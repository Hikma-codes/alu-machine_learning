#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    Attributes:
        nx(int): The number of input features to the neuron.

    Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.

    - public instance attributes
    """
    def __init__(self, nx):
        """
        Initializes the Neuron.

        Args:
            nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        # Initialize weight vector with random normal distribution
        self.W = np.random.normal(size=(1, nx))
        self.b = 0  # The bias
        self.A = 0  # The activated output of the neuron(prediction)
