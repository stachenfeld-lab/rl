""" This module contains the policy classes for the reinforcement learning algorithms

"""

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from torch.distributions import Categorical, MultivariateNormal, Normal


class DiscretePolicy(nn.Module):
    """
    Discrete policy network that selects actions from a categorical distribution
    Attributes:
        policy: Network used to sample actions
        nonlin: Activation function for the policy network
    Methods:
        sample: Sample an action from the policy
        act: Sample an action from the policy
    """

    def __init__(self, act_dim: int, hidden_size: int, activation: nn.Module = None):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        else:
            assert callable(activation), "Activation must be a torch.nn.functional"

        self.nonlin = activation
        self.policy = nn.Linear(hidden_size, act_dim)

    def sample(
        self, memory: torch.Tensor, tau: float = 1.0, action: torch.Tensor = None
    ):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :param tau: Temperature parameter for the Gumbel softmax
        :param action: Provided action to calculate the log probability of
        :return: action, action_probs, log_probs
        """
        action_logits = self.nonlin(self.policy(memory))

        action_probs = F.gumbel_softmax(action_logits, tau=tau, hard=True)
        action_dist = Categorical(action_probs)

        if action is None:
            action = action_dist.sample()

        log_probs = action_dist.log_prob(action)
        return action, log_probs, action_dist

    def act(self, memory: torch.Tensor, tau: float = 1.0):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :param tau: Temperature parameter for gumbel softmax
        :return: sampled action
        """
        action_logits = self.nonlin(self.policy(memory))
        action_probs = F.gumbel_softmax(action_logits, tau=tau, hard=True)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        return action

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network that selects actions from a normal distribution with a fixed standard deviation
    Attributes:
        policy: Network used to sample actions
        nonlin: Activation function for the policy network
        log_std: Log of the standard deviation for the normal distribution
        act_dim: Dimension of the action space
    Methods:
        sample: Sample an action from the policy
        act: Sample an action from the policy
    """

    def __init__(
        self, act_dim: int, hidden_size: int, activation: nn.functional = None
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        else:
            assert callable(activation), "Activation must be a torch.nn.functional"

        self.nonlin = activation
        self.policy = nn.Linear(hidden_size, act_dim)
        cov_matrix = torch.eye(act_dim) + torch.finfo(torch.float32).eps
        self.log_std = nn.Parameter(torch.log(1e-1 * cov_matrix))
        self.act_dim = act_dim

    def sample(self, memory: torch.Tensor, action: torch.Tensor = None):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :param action: Provided action to calculate the log probability of
        :return: action, action_probs, log_probs
        """
        action_mean = self.nonlin(self.policy(memory))
        action_std = torch.exp(self.log_std)
        action_dist = MultivariateNormal(action_mean, action_std)
        if action is None:
            action = action_dist.sample()

        log_probs = action_dist.log_prob(action)

        return action, log_probs, action_dist

    def act(self, memory: torch.Tensor):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :return: sampled action
        """
        action_mean = self.nonlin(self.policy(memory))
        action_std = torch.exp(self.log_std)
        action_dist = MultivariateNormal(action_mean, action_std)
        action = action_dist.sample()

        return action

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class VariableGaussianPolicy(nn.Module):
    """
    Gaussian policy network that selects actions from a normal distribution with a variable standard deviation
    Attributes:
        policy: Network used to sample actions
        nonlin: Activation function for the policy network
        log_std: Log of the standard deviation for the normal distribution
        act_dim: Dimension of the action space
    Methods:
        sample: Sample an action from the policy
        act: Sample an action from
    """

    def __init__(self, act_dim: int, hidden_size: int, activation: nn.Module = None):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        else:
            assert callable(activation), "Activation must be a torch.nn.functional"

        self.nonlin = activation
        self.policy = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        self.act_dim = act_dim

    def sample(self, memory: torch.Tensor, action: torch.Tensor = None):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :param action: Provided action to calculate the log probability of
        :return:
        """
        action_mean = self.nonlin(self.policy(memory))
        action_std = torch.diag_embed(torch.exp(self.log_std(memory)))
        action_dist = MultivariateNormal(action_mean, action_std)
        if action is None:
            action = action_dist.sample()
        log_probs = action_dist.log_prob(action).view(-1, 1)

        return action, log_probs, action_dist

    def act(self, memory: torch.Tensor):
        """
        Sample an action from the policy
        :param memory: Output of the actor network
        :return:
        """
        action_mean = self.nonlin(self.policy(memory))
        action_std = torch.diag_embed(torch.exp(self.log_std(memory)))
        action_dist = MultivariateNormal(action_mean, action_std)
        action = action_dist.sample()

        return action

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class Policies(Enum):
    DISCRETE = DiscretePolicy
    GAUSSIAN = GaussianPolicy
    VARIABLE_GAUSSIAN = VariableGaussianPolicy
