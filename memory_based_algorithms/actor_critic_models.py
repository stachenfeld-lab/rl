import pdb

import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from deep_models.feedforward_models import Feedforward
from algorithms.policies import Policies


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

        self.actor: nn.Module = Feedforward(
            obs_dim,
            hidden_size,
            [hidden_size, hidden_size],
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

    def _initialize_optimizers(self, actor_lr: float = 1e-4, critic_lr: float = 1e-3):
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

    def act(self, obs: torch.Tensor):
        """
        Sample an action from the policy
        :param obs: Observation from the environment [batch_size, obs_dim]
        :return: Sampled actions [batch_size, act_dim]
        """
        actor_output = self.actor(obs)
        return self.policy.act(actor_output)

    def get_value(self, obs: torch.Tensor, act: torch.Tensor):
        """
        Compute the Q-values for the given observations and actions
        :param obs: Observation from the environment [batch_size, obs_dim]
        :param act: Actions taken by the agent [batch_size, act_dim]
        :return: Q-values for the given observations and actions [batch_size, 1]
        """
        memory_representation = self.actor(obs)

        critic_input = torch.cat([memory_representation, act], dim=1)
        return self.critic1(critic_input), self.critic2(critic_input)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Compute the Q-values for the given observations and actions
        :param obs: Observation from the environment [batch_size, obs_dim]
        :param action: Provided action if one exists [batch_size, act_dim]
        :return: Q-values for the given observations and actions [batch_size, 1]
        """

        memory_representation = self.actor(obs)
        action, log_probs, action_distribution = self.policy(
            memory_representation, action=action
        )

        action_entropy = action_distribution.entropy()
        critic_input = torch.cat([memory_representation, action], dim=1)
        action_value1 = self.critic1(critic_input)
        action_value2 = self.critic2(critic_input)
        return action, action_value1, action_value2, log_probs, action_entropy

