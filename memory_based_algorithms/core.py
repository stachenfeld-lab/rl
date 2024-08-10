""" This module contains the core classes for the memory-based algorithms

"""

import numpy as np
import torch


def combined_shape(length, shape=None):
    """
    Combine the length and shape of the buffer
    :param length:
    :param shape:
    :return:
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    Replay buffer to store the experiences of the agent

    Attributes:
        obs_dim: Size of the observation space
        act_dim: Size of the action space
        size: Size of the replay buffer
        hidden_size: Size of the rnn hidden state
        device: Device to run the model on
        exploration_batch: Choose sampling episodes or time-steps
        exploitation_batch: Choose sampling episodes or time-steps
        obs_buf: Buffer to store the observations
        act_buf: Buffer to store the actions
        rew_buf: Buffer to store the rewards
        next_obs_buf: Buffer to store the next observations
        done_buf: Buffer to store the done flag
        hidden_buf: Buffer to store the rnn hidden state
        next_hidden_buf: Buffer to store the next rnn hidden state
        prev_act_buf: Buffer to store the previous actions
        prev_rew_buf: Buffer to store the previous rewards
        position: Current number of items in the replay buffer
        start_idx: Index to start sampling from
        max_size: Maximum size of the replay buffer

    Methods:
        store: Store the experiences in the replay buffer
        get: Get the experiences from the replay buffer
        reset: Reset the replay buffer
        finish_path: Finish the current
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        hidden_size: int,
        device: torch.device,
    ):
        """
        Initialize the replay buffer
        :param obs_dim: Size of the observation space
        :param act_dim: Size of the action space
        :param size: Size of the replay buffer
        :param hidden_size: Size of rnn hidden state
        :param device: Device to run the model on
        """

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.hidden_size = hidden_size
        self.device = device

        # choose sampling episodes or time-steps
        self.exploration_batch = np.array([])
        self.exploitation_batch = np.array([])

        # initialize buffers
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.hidden_buf = np.zeros(combined_shape(size, hidden_size), dtype=np.float32)
        self.next_hidden_buf = np.zeros(
            combined_shape(size, hidden_size), dtype=np.float32
        )
        self.prev_act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.prev_rew_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)

        self.position, self.start_idx, self.max_size = 0, 0, size

    def store(
        self, obs, next_obs, act, rew, done, prev_act, prev_rew, hidden, next_hidden
    ):
        """
        Add one timestep of agent experience to the replay buffer
        :param obs: Observation from the environment
        :param next_obs: Next observation from the environment
        :param act: Action taken by the agent
        :param rew: Reward received by the agent
        :param done: Done flag
        :param prev_act: Previous action taken by the agent
        :param prev_rew: Previous reward received by the agent
        :param hidden: Hidden state of the rnn
        :param next_hidden: Next hidden state of the rnn
        """

        assert self.position <= self.max_size, "Buffer overflow"

        # store the experiences in the replay buffer
        self.obs_buf[self.position] = obs
        self.next_obs_buf[self.position] = next_obs
        self.act_buf[self.position] = act
        self.rew_buf[self.position] = rew
        self.done_buf[self.position] = done
        self.prev_act_buf[self.position] = prev_act
        self.prev_rew_buf[self.position] = prev_rew
        self.hidden_buf[self.position] = hidden
        self.next_hidden_buf[self.position] = next_hidden

        # update the capacity
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self, batch_size: int = None):
        """
        Get all experiences from the replay buffer
        :param batch_size:
        :return: shuffled batch of experiences
        """

        if batch_size is not None:
            np.random.shuffle(self.exploitation_batch)
            return np.random.choice(self.exploitation_batch, batch_size)

        np.random.shuffle(self.exploitation_batch)
        return self.exploitation_batch

    def reset(self):
        """
        Reset the replay buffer
        :return:
        """

        self.obs_buf = np.zeros_like(self.obs_buf)
        self.next_obs_buf = np.zeros_like(self.next_obs_buf)
        self.act_buf = np.zeros_like(self.act_buf)
        self.rew_buf = np.zeros_like(self.rew_buf)
        self.done_buf = np.zeros_like(self.done_buf)
        self.prev_act_buf = np.zeros_like(self.prev_act_buf)
        self.prev_rew_buf = np.zeros_like(self.prev_rew_buf)
        self.hidden_buf = np.zeros_like(self.hidden_buf)
        self.next_hidden_buf = np.zeros_like(self.next_hidden_buf)

        self.exploitation_batch, self.exploration_batch = np.array([]), np.array([])
        self.position, self.start_idx = 0, 0
        raise NotImplementedError

    def finish_path(self):
        path_slice = slice(self.start_idx, self.position)

        # fill the exploration buffer
        data = dict(
            obs=self.obs_buf[path_slice],
            next_obs=self.next_obs_buf[path_slice],
            act=self.act_buf[path_slice],
            rew=self.rew_buf[path_slice],
            done=self.done_buf[path_slice],
            prev_act=self.prev_act_buf[path_slice],
            prev_rew=self.prev_rew_buf[path_slice],
            hidden=self.hidden_buf[self.start_idx],
            next_hidden=self.next_hidden_buf[self.start_idx],
        )

        # append the data to the exploration buffer
        data = {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in data.items()
        }
        self.exploitation_batch = np.append(self.exploitation_batch, data)
