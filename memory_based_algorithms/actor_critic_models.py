import pdb

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import chain
from deep_models.feedforward_models import Feedforward
from deep_models.sequential_models import RNN
from algorithms.policies import Policies
from memory_based_algorithms.core import ReplayBuffer


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
        activation: nn.Module = None,
        policy_type: str = "discrete",
        policy_kwargs: dict = None,
        device: torch.device = None,
        optimizer_kwargs: dict = None,
    ):
        super().__init__()

        # set default device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        if policy_kwargs is None:
            policy_kwargs = {}

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if activation is None:
            activation = nn.ReLU()
        else:
            assert callable(activation), "Activation must be a torch.nn.functional"

        self.actor: nn.Module = RNN(
            obs_dim,
            hidden_size,
            activation,
            device=self.device,
        )

        self.critic1: nn.Module = Feedforward(
            hidden_size + act_dim,
            1,
            [hidden_size, hidden_size],
            activation,
            device=self.device,
        )

        self.critic2: nn.Module = Feedforward(
            hidden_size + act_dim,
            1,
            [hidden_size, hidden_size],
            activation,
            device=self.device,
        )

        self.policy = Policies[policy_type.upper()].value(
            act_dim, hidden_size, activation, **policy_kwargs
        )

        self.policy = self.policy.to(self.device)

        # initialize the optimizers
        self._initialize_optimizers(**optimizer_kwargs)

    def _initialize_optimizers(self, actor_lr: float = 1e-3, critic_lr: float = 1e-3):
        """
        Initialize the optimizers for the actor and critic networks
        :param actor_lr: Learning rate for the actor network
        :param critic_lr: Learning rate for the critic network
        :return: None
        """
        critic_chain = chain(self.critic1.parameters(), self.critic2.parameters())
        actor_chain = chain(self.actor.parameters(), self.policy.parameters())

        self.optimizer = {
            "actor": torch.optim.Adam(actor_chain, lr=actor_lr),
            "critic": torch.optim.Adam(critic_chain, lr=critic_lr),
        }

    def act(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        """
        Sample an action from the policy
        :param obs: Observation from the environment [batch_size, obs_dim]
        :param hidden_state: Hidden state of the RNN [batch_size, hidden_size]
        :return: Sampled actions [batch_size, act_dim]
        """
        actor_output, actor_previous = self.actor(obs, hidden_state)
        return self.policy.act(actor_output), actor_output, actor_previous

    def get_obs_value(
        self, obs: torch.Tensor, act: torch.Tensor, hidden_state: torch.Tensor = None
    ):
        """
        Compute the Q-values for the given observation and action starting from a particular observation.
        :param obs: Observation from the environment [batch_size, obs_dim]
        :param act: Action taken by the agent [batch_size, act_dim]
        :param hidden_state: Hidden state of the RNN [batch_size, hidden_size]
        :return: Q-values for the given observations and actions [batch_size, 1]
        """
        memory_representation, _ = self.actor(obs, hidden_state)
        critic_input = torch.cat((memory_representation, act), dim=-1)
        return self.critic1(critic_input), self.critic2(critic_input)

    def get_mem_value(self, memory_representation: torch.Tensor, act: torch.Tensor):
        """
        Compute the Q-values for the given memory state and action of the network.
        :param memory_representation: Memory representation of the observations [batch_size, hidden_size]
        :param act: Action taken by the agent [batch_size, act_dim]
        :return: Q-values for the given observations and actions [batch_size, 1]
        """
        critic_input = torch.cat((memory_representation, act), dim=-1)
        return self.critic1(critic_input), self.critic2(critic_input)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor = None,
        action: torch.Tensor = None,
    ):
        """
        Compute the Q-values for the given observations and actions
        :param obs: Observation from the environment [batch_size, obs_dim]
        :param hidden_state: Hidden state of the RNN [batch_size, hidden_size]
        :param action: Provided action if one exists [batch_size, act_dim]
        :return: Q-values for the given observations and actions [batch_size, 1]
        """

        memory_representation, _ = self.actor(obs, hidden_state)
        action, log_probs, action_distribution = self.policy(
            memory_representation, action=action
        )

        action_entropy = action_distribution.entropy()
        critic_input = torch.cat((memory_representation, action), dim=-1)
        action_value1 = self.critic1(critic_input)
        action_value2 = self.critic2(critic_input)

        return (
            memory_representation,
            action,
            action_value1,
            action_value2,
            log_probs,
            action_entropy,
        )


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
        policy_type: str = "discrete",
        optimization_kwargs: dict = None,
        buffer_kwargs: dict = None,
        buffer_size: int = 1000,
        batch_size: int = 10,
        seed: int = None,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__()

        # set default device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.hidden_size = hidden_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor_critic = ActorCritic(
            obs_dim,
            act_dim,
            hidden_size,
            policy_type=policy_type,
            device=device,
            **kwargs,
        )

        self.target_actor_critic = deepcopy(self.actor_critic)

        # freeze the target networks
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False

        # initialize the replay buffer
        self._initialize_buffer(
            obs_dim, act_dim, hidden_size, buffer_size, batch_size, device
        )

    def _initialize_buffer(
        self, obs_dim, act_dim, hidden_size, size, batch_size, device
    ):
        """
        Initialize the replay buffer
        :param obs_dim: Dimension of the observation space
        :param act_dim: Dimension of the action space
        :param size: Size of the replay buffer
        :param hidden_size: Size of the hidden state
        :param batch_size: Size of the batch
        :param device: Device to run the model on
        :return: None
        """
        self.replay_buffer = ReplayBuffer(
            obs_dim, act_dim, size, batch_size, hidden_size, device
        )

    def compute_critic_loss(self, batch):
        """
        Compute the loss for the critic network
        :param batch: Batch of experiences
        :return: Loss for the critic network
        """
        obs, act, rew, next_obs, done = (
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["next_obs"],
            batch["done"],
        )
        next_hidden = batch["next_hidden"]

        q1, q2 = self.actor_critic.get_mem_value(next_hidden, act)

        q1 = q1.squeeze()
        q2 = q2.squeeze()

        # sample actions from current policy
        with torch.no_grad():

            memory_embedding = torch.zeros(
                (
                    self.replay_buffer.position,
                    self.replay_buffer.batch_size,
                    self.hidden_size,
                ),
                device=self.device,
            )
            next_actions = torch.zeros_like(batch["act"])
            next_action_log_probs = torch.zeros(
                (self.replay_buffer.position, self.replay_buffer.batch_size),
                device=self.device,
            )
            entropy = torch.zeros(
                (self.replay_buffer.position, self.replay_buffer.batch_size),
                device=self.device,
            )

            target_q1 = torch.zeros(
                (self.replay_buffer.position, self.replay_buffer.batch_size, 1),
                device=self.device,
            )

            target_q2 = torch.zeros(
                (self.replay_buffer.position, self.replay_buffer.batch_size, 1),
                device=self.device,
            )

            # propagate initial hidden state
            for ti in range(self.replay_buffer.position):
                (
                    memory_embedding[ti],
                    next_actions[ti],
                    _,
                    _,
                    next_action_log_probs[ti],
                    entropy[ti],
                ) = self.actor_critic.get_action_and_value(
                    next_obs[ti],
                    action=act[ti],
                    hidden_state=next_hidden[ti],
                )

                target_q1[ti], target_q2[ti] = self.target_actor_critic.get_obs_value(
                    next_obs[ti], next_actions[ti], memory_embedding[ti]
                )

            self.alpha = 1
            self.gamma = 0.9

            target_q = torch.min(target_q1, target_q2).squeeze()
            next_q = self.gamma * target_q - self.alpha * next_action_log_probs

        backup = rew + (1 - done) * next_q
        # compute loss
        q1_loss = (q1 - backup).pow(2).mean()
        q2_loss = (q2 - backup).pow(2).mean()
        q_loss = q1_loss + q2_loss

        loss_info = {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item()}

        return q_loss, loss_info

    def compute_policy_loss(self, batch):
        """
        Compute the loss for the policy network
        :param batch: Batch of experiences
        :return: Loss for the policy network
        """
        obs, prev_act, prev_rew = (
            batch["obs"],
            batch["prev_act"],
            batch["prev_rew"],
        )

        mem_initial = batch["hidden"]
        memory_representation = torch.zeros(
            (
                self.replay_buffer.position,
                self.replay_buffer.batch_size,
                self.hidden_size,
            ),
            device=self.device,
        )

        # propagate initial hidden state to get memory representation
        for step in range(self.replay_buffer.position):
            memory_representation[step], _ = self.actor_critic.actor(
                obs[step], (mem_initial if step == 0 else None)
            )

        actions, log_probs, action_distributions = self.actor_critic.policy(
            memory_representation
        )

        # get Q-values
        with torch.no_grad():
            q1, q2 = self.actor_critic.get_mem_value(memory_representation, actions)

            q_values = torch.min(q1, q2).squeeze()

        # get expected Q
        expected_q = (q_values * log_probs).sum(dim=1, keepdim=True)

        # compute loss
        policy_loss = (-expected_q - self.alpha * action_distributions.entropy()).mean()

        loss_info = {
            "policy_loss": policy_loss.item(),
            "entropy": action_distributions.entropy().mean().item(),
        }
        return policy_loss, loss_info

    def update(self, batch_size: int = 10):
        """
        Update the actor critic networks
        :return:
        """

        # sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.get(batch_size)
        c_loss, p_loss = [], []
        for episode in batch:
            # compute the loss for the critic network
            critic_loss, critic_info = self.compute_critic_loss(episode)

            # performance optimization step on critic
            self.actor_critic.optimizer["critic"].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1)
            self.actor_critic.optimizer["critic"].step()

            # compute the loss for the policy network
            # freeze the critic networks
            for param in chain(
                self.actor_critic.critic1.parameters(),
                self.actor_critic.critic2.parameters(),
            ):
                param.requires_grad = False

            policy_loss, policy_info = self.compute_policy_loss(episode)

            # performance optimization step on policy
            self.actor_critic.optimizer["actor"].zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1)
            self.actor_critic.optimizer["actor"].step()
            p_loss.append(policy_loss.item())
            c_loss.append(critic_loss.item())

            # unfreeze the critic networks
            for param in chain(
                self.actor_critic.critic1.parameters(),
                self.actor_critic.critic2.parameters(),
            ):
                param.requires_grad = True

                # update the target networks with polyak averaging
            with torch.no_grad():
                for param, target_param in zip(
                    self.actor_critic.parameters(),
                    self.target_actor_critic.parameters(),
                ):
                    # w_targ = rho * w_targ + (1 - rho) * w
                    rho = 0.995
                    target_param.data.mul_(rho)
                    target_param.data.add_((1 - rho) * param.data)

        print(f"policy loss: {np.mean(p_loss)}; critic loss: {np.mean(c_loss)}")
        self.replay_buffer.reset()

    def sample_episodes(
        self, obs_map: torch.Tensor, observations: torch.Tensor, steps=50
    ):

        n_env = self.replay_buffer.batch_size
        self.actor_critic.actor.initialize_hidden(n_env)
        a2, r2 = 0, 0
        rewards = []
        for epoch in range(steps):
            # get observation state
            obs = observations[epoch]

            # get action and hidden states
            action, hidden, hidden_prev = self.actor_critic.act(obs)

            # get reward
            with torch.no_grad():
                reward = -torch.sqrt((action - obs @ obs_map) ** 2).sum(dim=1).squeeze()

            rewards.append(reward)

            # get next observation state
            next_obs = observations[epoch + 1]

            # get done
            done = torch.zeros(n_env)

            # store experience in the replay buffer
            self.replay_buffer.store(
                obs,
                next_obs,
                action,
                reward,
                done,
                a2,
                r2,
                hidden.detach(),
                hidden_prev.detach(),
            )
            a2 = action
            r2 = reward
        # print(f"Episode Average reward: {torch.stack(rewards).mean().numpy()}.")
        self.replay_buffer.finish_path()


Model = SoftActorCritic(
    4,
    2,
    32,
    device=torch.device("cpu"),
    policy_type="gaussian",
    seed=1,
    batch_size=25,
    buffer_size=250,
)

reward_predictor = torch.randn(2, 1)
obs_mapping = torch.randn(4, 2)
steps = 50
episode_steps = 50
observations = torch.randn((episode_steps + 1, 25, 4))
for _ in range(steps):
    Model.sample_episodes(obs_mapping, observations)
    Model.update()
pdb.set_trace()
