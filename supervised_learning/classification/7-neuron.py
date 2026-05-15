#!/usr/bin/env python3
"""
Class Neuron that defines a single neuron performing binary classification
"""


import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X(numpy.ndarray): Shape(nx, m) contains input data
                m: number of examples

        Returns:
            Neuron's prediction
            Cost function
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A > 0.5, 1, 0)
        return (pred, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent

        Args:
            X(numpy.ndarray)
            Y(numpy.ndarray)
            A(numpy.ndarray)
            alpha: learning rate

        Updates the private attributes __W and __b
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1/m) * np.matmul(dz, X.T)
        db = np.mean(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron by updating the __W, __b, and __A

        Args:
            X: input data
            Y: Correct labels of the input data
            iterations: Number of iterations to train over
            alpha: Learning rate
            verbose (bool, optional): _description_. Defaults to True.
            graph (bool, optional): _description_. Defaults to True.
            step (int, optional): _description_. Defaults to 100.

        Returns:
            The evaluation of the training data after
            iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        costs = []
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print('Cost after {} iterations: {}'.format(i, cost))
            if graph and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
        if graph and costs:
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
