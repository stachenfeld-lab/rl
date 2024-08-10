import pdb

import numpy as np
import torch
import torch.nn as nn
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
            obs_dim + act_dim,
            1,
            [hidden_size, hidden_size],
            activation,
            device=self.device,
        )

        self.critic2: nn.Module = Feedforward(
            obs_dim + act_dim,
            1,
            [hidden_size, hidden_size],
            activation,
            device=self.device,
        )

        self.policy = Policies[policy_type.upper()].value(
            act_dim, hidden_size, activation, **policy_kwargs
        )

        self.policy = self.policy.to(self.device)

    def act(self, obs: torch.Tensor):
        """
        Sample an action from the policy
        :param obs: Observation from the environment [batch_size, obs_dim]
        :return: Sampled actions [batch_size, act_dim]
        """
        actor_output = self.actor(obs)
        return self.policy.act(actor_output)
