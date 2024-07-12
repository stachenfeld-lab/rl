"""This module contains classes for various algorithms.

Typical usage example:


"""
import abc
import pdb
import random
import time
from typing import Optional, Tuple
from datetime import datetime

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter


class RlAlgorithm(nn.Module, metaclass=abc.ABCMeta):
    """Base class for RL algorithms

    Attributes:
        action_space (gymnasium):
            The action space of the environment
        alpha_entropy:
            Weight of entropy regularization term
        distribution:
            Action distribution used for action generation.
            Can be a discrete or continuous distribution.
        gamma:
            Future reward discount factor.
        seed:
            Seed used for random number generation
        num_envs:
            Number of parallel environments
        num_steps:
            Number of steps per environment
        num_mini_batches:
            Number of minibatches
        num_update_epochs:
            Number of gradient steps per rollout episode
        max_grad_norm:
            Maximum gradient norm used for gradient clipping
        total_time_steps:
            Maximum number of environment steps used for training.
        anneal_lr:
            Whether to anneal learning rates.
        gym_id:
            Name of gym environment
        device:
            Torch device to run model on
        batch_size:
            Model batch size
        observation_dimension:
            Dimension of the observation space
        action_dimension:
            Dimension of the action space
        env:
            Vectorized gymnasium environment.
        writer:
            Tensorboard writer
    """

    def __init__(
        self,
        action_space: str = "continuous",
        summary_writer: Optional[SummaryWriter] = None,
        capture_videos: bool = True,
        **hyperparameters,
    ):
        """Initialize the class with hyperparameters"""

        assert action_space in ["discrete", "continuous"]

        super(RlAlgorithm, self).__init__()
        self.__init_hyperparameters(**hyperparameters)

        # Set manual seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if action_space == "continuous":
            self.distribution = MultivariateNormal
        else:
            self.distribution = Categorical
        self.action_space = action_space
        self.capture_videos = capture_videos

        # Make vectorized environments

        env = gymnasium.vector.SyncVectorEnv(
            [
                self.make_env(self.gym_id, idx, self.seed, capture_videos)
                for idx in range(self.num_envs)
            ]
        )

        if summary_writer is None:
            run_name = (
                f"{hyperparameters['gym_id']}__{hyperparameters['exp_name']}"
                f"__{hyperparameters['seed']}__{int(time.time())}"
            )
            summary_writer = SummaryWriter(f"runs/{run_name}")
            summary_writer.add_text(
                "Hyperparameters",
                f"|param|value|\n|-|-\n%s"
                % (
                    "\n".join(
                        [f"|{key}|{value}|" for key, value in hyperparameters.items()]
                    )
                ),
            )
        self.writer = summary_writer
        self.env = env
        self.observation_dimension = np.prod(env.single_observation_space.shape)

        if action_space == "discrete":
            self.action_dimension = int(np.prod(env.single_action_space.n))
        else:
            self.action_dimension = int(np.prod(env.single_action_space.shape))

    def __init_hyperparameters(
        self,
        gamma: float = 0.99,
        seed: int = 1,
        num_envs: int = 4,
        num_steps: int = 128,
        num_mini_batches: int = 4,
        num_update_epochs: int = 4,
        action_std: float = 0.1,
        alpha_entropy: float = 1e-2,
        max_grad_norm: float = 1.0,
        total_time_steps: int = 25000,
        anneal_lr: bool = True,
        learn_exploration: bool = True,
        gym_id: str = "CartPole-v1",
        device: torch.device = None,
        **kwargs,
    ):
        """Initialize algorithm hyperparameters

        Args:
            gamma:
                Future reward discount factor.
            seed:
                Seed used for random number generation
            num_envs:
                Number of parallel environments
            num_steps:
                Number of steps per environment
            num_mini_batches:
                Number of minibatches
            num_update_epochs:
                Number of gradient steps per rollout episode
            action_std:
                Standard deviation of the selected actions
            alpha_entropy:
                Weight of entropy regularization term
            max_grad_norm:
                Maximum gradient norm used for gradient clipping
            total_time_steps:
                Maximum number of environment steps used for training.
            anneal_lr:
                Whether to anneal learning rates.
            learn_exploration:
                Whether to make the level of action noise learnable.
            gym_id:
                Name of gym environment
            **kwargs:

        Returns:

        """

        self.gamma = gamma
        self.seed = seed
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_mini_batches = num_mini_batches
        self.num_update_epochs = num_update_epochs
        self.alpha_entropy = alpha_entropy
        self.max_grad_norm = max_grad_norm
        self.total_time_steps = total_time_steps
        self.anneal_lr = anneal_lr
        self.gym_id = gym_id
        self.action_std_fixed = action_std
        self.learn_exploration = learn_exploration

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        self.batch_size = int(self.num_steps * self.num_envs)

    @staticmethod
    def make_env(
        gym_id, idx, seed, capture_videos: bool = True, recording_frequency: int = 100
    ):
        """Initialize the vecortized environments

        Args:
            gym_id:
            idx:
            seed:
            capture_videos:
            recording_frequency:

        Returns:
            thunk:
        """

        def thunk():
            env = gymnasium.make(gym_id, render_mode="rgb_array")
            env = gymnasium.wrappers.RecordEpisodeStatistics(env)
            if capture_videos:
                date = datetime.today().strftime("%Y-%m-%d")
                if idx == 0:  # record actions of the first environment only
                    env = gymnasium.wrappers.RecordVideo(
                        env,
                        f"videos/{gym_id}/{date}",
                        step_trigger=lambda t: (t % recording_frequency) == 0,
                    )
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    def learning(self):
        """Simulates training loop

        Returns:

        """


if __name__ == '__main__':
    pass
