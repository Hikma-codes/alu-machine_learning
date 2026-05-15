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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X(numpy.ndarray): Contains input data with shape(nx, m):
                -nx: Number of input features to the neuron
                -m: The number of examples

        Updates the private attribute __A
        Neuron uses a sigmoid activation function and returns the __A attribute
        """

        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y(numpy.ndarray): Shape(1, m) contains correct labels for the input
            A(numpy.ndarray): Shape(1, m) contains the activated output

        Returns:
            Cost function
        """
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost
