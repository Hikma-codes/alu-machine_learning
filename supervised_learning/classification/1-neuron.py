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
        __W(numpy.ndarray): The weight vector of the neuron
        __b(float): The bias of the neuron
        __A(float): The activated output of the neuron

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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0  # Private bias of the neuron
        self.__A = 0  # Private activated output of the neuron(prediction)

    # Getter functions
    @property
    def W(self):
        """
        Getter of the weight vector
        """
        return self.__W

    @property
    def b(self):
        """
        Getter of the bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter of the activated output
        """
        return self.__A
