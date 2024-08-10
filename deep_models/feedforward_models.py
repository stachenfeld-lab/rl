""" This file contains feedforward deep neural network architectures
 to be used for rl problems later
example:
    input_size, batch_size, output_size = 4, 10, 2
    model = Feedforward(input_size=input_size, output_size=output_size, hidden_layer_sizes=[10, 10],
    activation_function=nn.Tanh, output_activation=nn.Identity)

    inputs = torch.randn(batch_size, input_size)

"""

import pdb
from typing import Sequence

import torch
import torch.nn as nn


class Feedforward(nn.Module):
    """Implements a basic feedforward neural network with a
     variable number of hidden layers and variable activation
     functions.

    Args:
        input_size: Size of the inputs to the model
        hidden_layer_sizes: Sequence of integers representing the size of each hidden layer
        output_size: Size of the output of the model
        activation_function: Activation function to use for hidden layers
        output_activation: Activation function to use for the final output layer
        **kwargs:
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: Sequence = None,
        activation_function: nn.functional = None,
        output_activation: nn.functional = None,
        device: torch.device = None,
        **kwargs
    ):
        super(Feedforward, self).__init__()
        # define default activations
        if activation_function is None:
            activation_function = nn.ReLU()

        if output_activation is None:
            output_activation = nn.Identity()

        # if hidden_layer_sizes is None
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50]

        # store the input size, hidden layer sizes, and output size
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # set default device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # construct network architecture

        modules = []
        previous_size = input_size

        for entry, size in enumerate(hidden_layer_sizes):
            modules.append(nn.Linear(previous_size, size))
            modules.append(activation_function)
            previous_size = size

        modules.append(nn.Linear(previous_size, output_size))
        modules.append(output_activation)

        self.model = nn.Sequential(*modules)

        self.model = self.model.to(self.device)

    def forward(self, x):
        y = self.model(x)
        return y


if __name__ == "__main__":
    pass
