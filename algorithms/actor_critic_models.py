"""This module contains classes for various algorithms.

Typical usage example:


"""
import abc
import ipdb as pdb
import random
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from algorithms.core import RlAlgorithm
from typing_extensions import override
from deep_models.feedforward_models import Feedforward


class BaseActorCritic(RlAlgorithm):
    """Base class for actor-critic based RL methods

    Attributes:
        actor:
            Actor-Network used for action selection
        actor_std:
            Standard deviation of actions
        actor_lr:
            Learning rate used for the actor network
        critic:
            Critic-Network used for evaluating actions.
        critic_lr:
            Learning rate used for the actor network
        optimizers:
            Optimizers used for training the actor and critic networks

    """

    def __init__(
        self,
        actor: Optional[nn.Module] = None,
        critic: Optional[nn.Module] = None,
        **hyperparams,
    ):
        """Initialize the base actor-critic class"""

        super(BaseActorCritic, self).__init__(**hyperparams)
        if actor is None:
            self.actor = Feedforward(
                input_size=self.observation_dimension,
                hidden_layer_sizes=(500, ),
                output_size=self.action_dimension,
                activation_function=nn.Tanh,
                std=1,
            ).to(self.device)
        else:
            self.actor = actor.to(self.device)

        if self.action_space == "continuous":
            if self.learn_exploration:
                self.actor_std = nn.Parameter(
                    torch.log(torch.tensor(self.action_std_fixed))
                    * torch.ones(1, self.action_dimension).to(self.device)
                )
            else:
                self.actor_std = self.action_std_fixed
        else:
            self.actor_std = None

        if critic is None:
            self.critic = Feedforward(
                input_size=self.observation_dimension,
                layer_sizes=(64, 64),
                output_size=1,
                activation_function=nn.Tanh,
                std=1,
            ).to(self.device)
        else:
            self.critic = critic.to(self.device)

        self._initialize_optimizers(**hyperparams)

    def _initialize_optimizers(self, actor_lr: float = 1e-4, critic_lr: float = 1e-3, **hyperparams):
        """Initialize the actor and critic optimizer params
        Args:
            actor_lr (float):
                Actor learning rate
            critic_lr (float):
                Critic learning rate
        Returns:

        """
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=actor_lr),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=critic_lr),
        }

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def get_value(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the value for the given observation/state
        Args:
            observation: Current observation/state
        Returns:
            value: Current value for the given observation/state
        """

        value = self.critic(observation)
        return value

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """Returns the action and value for the given observation

        Args:
            observation:
                Current observation/state
            action:
                Optional sampled action. If given then this returns the log-probs
                and statistics for the given action. Otherwise, it samples an
                action and returns the log-probs.

        Returns:
            action:
                Sampled action from the actor distribution.
            action_log_probs:
                Log probability of the sampled actions.
            action_entropy:
                Entropy of the action distribution
            observation_value:
                Estimated value of the current observation/state
        """
        previous_state = None
        if self.actor_std is None:
            action_logits = self.actor(observation)
            action_probs = self.distribution(action_logits)
        else:
            action_means = self.actor(observation)
            if self.learn_exploration:
                action_stds = self.actor_std.expand_as(action_means).exp()
                action_probs = self.distribution(
                    action_means, torch.diag_embed(action_stds)
                )
            else:
                action_probs = self.distribution(
                    action_means,
                    self.actor_std
                    * torch.eye(self.action_dimension, device=self.device),
                )

        if action is None:
            action = action_probs.sample()
        action_log_probs = action_probs.log_prob(action)
        action_entropy = action_probs.entropy()
        observation_value = self.critic(observation)

        return (
            action,
            action_log_probs,
            action_entropy,
            observation_value,
            previous_state,
        )


class PPO(BaseActorCritic):
    """Implementation of the PPO algorithm"""

    def __init__(self, **hyperparams):
        """Initialize the PPO algorithm with hyper-parameters"""
        super(PPO, self).__init__(**hyperparams)
        self._init_ppo_hyperparameters(**hyperparams)
        self.final_rewards = []

    def _init_ppo_hyperparameters(
        self,
        gae_lambda: float = 0.95,
        clip_coeff: float = 0.2,
        use_gae: bool = True,
        normalize_advantage: bool = True,
        clip_critic: bool = True,
        **kwargs,
    ):
        """

        Args:
            gae_lambda: Generalized advantage lambda coefficient, by default 0.95
            clip_coeff: PPO clipping coefficient, by default 0.2
            use_gae: A boolean flag that specifies whether to use a traditional advantage or the generalized advantage
                by default True
            normalize_advantage: Whether to normalize the advantages each roll out, by default True
            clip_critic: Whether to clip the critic objective function, by default True
            **kwargs:

        Returns:

        """

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.clip_critic = clip_critic
        self.clip_coeff = clip_coeff

    @override
    def learning(self):

        obs = torch.zeros(
            (self.num_steps, self.num_envs, self.observation_dimension)
        ).to(self.device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + self.env.single_action_space.shape
        ).to(self.device)

        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        n_updates = self.total_time_steps // self.batch_size
        global_step = 0
        start_time = time.time()

        next_obs = torch.Tensor(self.env.reset()[0]).to(self.device)
        next_done = torch.Tensor(self.num_envs).to(self.device)
        for update in range(1, n_updates + 1):
            if self.anneal_lr:
                frac = 1 - (update - 1) / n_updates
                new_lr = self.actor_lr * frac
                self.optimizers["actor"].param_groups[0]["lr"] = new_lr
            for step in range(self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    (
                        action,
                        logprob,
                        _,
                        value,
                        previous_state,
                    ) = self.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Step environment
                next_obs, reward, done, _, info = self.env.step(action.cpu().numpy())

                if len(done) > 1:
                    for bdone in done:
                        if bdone:
                            self.actor.reset_state(isolated_idx=bdone)
                            self.critic.reset_state(isolated_idx=bdone)
                elif done:
                    self.actor.reset_state(batch_size=1)
                    self.critic.reset_state(batch_size=1)

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                for key, value in info.items():
                    if key == "final_info":
                        for item in value:
                            if isinstance(item, dict) and "episode" in item.keys():
                                print(
                                    f"global_step: {global_step} Episode return: {item['episode']['r']}"
                                )
                                self.writer.add_scalar(
                                    "charts/episodic_return",
                                    item["episode"]["r"],
                                    global_step,
                                )
                                self.writer.add_scalar(
                                    "charts/episodic_length",
                                    item["episode"]["l"],
                                    global_step,
                                )

            with torch.no_grad():
                next_value = self.get_value(next_obs).reshape(1, -1)
                if self.use_gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + (self.gamma * self.gae_lambda) ** t
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.num_steps)):
                        if t == self.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + self.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            b_obs = obs.reshape((-1,) + self.env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)

            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            minibatch_size = self.batch_size // self.num_mini_batches
            for epoch in range(self.num_update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, new_values, _ = self.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                    )

                    log_ratio = newlogprob - b_logprobs[mb_inds]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl_approx = ((ratio - 1) - log_ratio).mean()
                    mb_advantages = b_advantages[mb_inds]

                    if self.normalize_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + torch.finfo().eps
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coeff, 1 + self.clip_coeff
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    new_value = new_values.view(-1)
                    if self.clip_critic:
                        v_loss_u = (new_value - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            new_value - b_values[mb_inds],
                            -self.clip_coeff,
                            self.clip_coeff,
                        )

                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_u, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.alpha_entropy * entropy_loss

                    self.optimizers["actor"].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm
                    )
                    self.optimizers["actor"].step()
                    self.optimizers["critic"].zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.optimizers["critic"].step()
                    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = (
                        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    )

                    self.writer.add_scalar(
                        "charts/learning_rate",
                        self.optimizers["actor"].param_groups[0]["lr"],
                        global_step,
                    )
                    self.writer.add_scalar(
                        "losses/value_loss", v_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/policy_loss", pg_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/entropy", entropy_loss.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/approx_kl", kl_approx.item(), global_step
                    )
                    self.writer.add_scalar(
                        "losses/explained_variance", explained_var, global_step
                    )
                    self.writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

        self.env.close()
        self.writer.close()

    def evaluate(self):
        for _ in range(25):
            next_obs = torch.Tensor(self.env.reset()[0]).to(self.device)
            self.final_rewards = []
            for step in range(self.num_steps):
                with torch.no_grad():
                    action, logprob, _, value, _ = self.get_action_and_value(next_obs)

                # Step environment

                next_obs, reward, done, _, info = self.env.step(action.cpu().numpy())
                if len(done) > 1:
                    for bdone in done:
                        if bdone:
                            self.actor.reset_state(isolated_idx=bdone)
                elif done:
                    self.actor.reset_state(batch_size=1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                for key, value in info.items():
                    if key == "final_info":
                        for item in value:
                            if isinstance(item, dict) and "episode" in item.keys():
                                self.final_rewards.append(item["episode"]["r"].mean())
        self.env.close()
        return np.mean(self.final_rewards)


if __name__ == '__main__':
    pass
