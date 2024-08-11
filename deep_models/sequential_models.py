""" This file contains sequential deep neural network architectures
 to be used for rl problems later
example:
    input_size, batch_size, output_size = 4, 10, 2

"""

import pdb

import numpy as np
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: nn.Module = None,
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

        if activation is None:
            activation = nn.ReLU()
        else:
            assert callable(activation), "Activation must be a torch.nn.functional"
        self.activation = activation

        self._initialize_weights(input_size, hidden_size)
        self.x = None
        self.initialize_hidden()

    def _initialize_weights(self, input_size: int, hidden_size: int, tau: float = 10):
        """
        Initialize the weights of the rnn
        :param input_size: Size of the input
        :param hidden_size: Size of the hidden state
        :param tau: Membrane time constant
        :return: None
        """
        self.hidden_size = hidden_size
        self.Input = nn.Parameter(
            torch.randn(input_size, hidden_size, device=self.device)
            / np.sqrt(input_size)
        )
        self.J = nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=self.device)
            / np.sqrt(hidden_size)
        )
        self.b = nn.Parameter(torch.zeros(hidden_size, device=self.device))
        self.tau = tau

    def initialize_hidden(self, batch_size: int = 16):
        """
        Initialize the hidden state of the rnn
        :param batch_size: Size of the batch
        :return: hidden state
        """
        x = torch.randn(batch_size, self.hidden_size) / np.sqrt(batch_size)
        self.x = x.to(self.device)

    def forward(
        self, inputs, hidden: torch.Tensor = None
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the rnn
        :param inputs: Inputs passed to the RNN [batch, input_size]
        :param hidden: Hidden state of the RNN [batch, hidden_size], if none, use self.x
        :return: Network output, network hidden state, network previous hidden state
        """
        if hidden is None:
            x0 = self.x
        else:
            x0 = hidden

        dx = -x0 + self.activation(x0) @ self.J + inputs @ self.Input + self.b
        x = x0 + dx / self.tau
        self.x = x

        return self.activation(x), self.activation(x0)
