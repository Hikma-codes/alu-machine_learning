#!/usr/bin/env python3
"""Defines a neural network"""

import numpy as np


class NeuralNetwork:
    """
    Class defines a neural network

    Args:
        nx: number of input features
        nodes: number of nodes found in the hidden layer
        W1: Weight vector of the hidden layer
        W2: Weight vector of the output neuron
        b1: Bias of the hidden layer
        b2: Bias of the output layer
        A1: Prediction of the hidden layer
        A2: Prediction of the output layer
    """
    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.nx = nx
        self.nodes = nodes
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
