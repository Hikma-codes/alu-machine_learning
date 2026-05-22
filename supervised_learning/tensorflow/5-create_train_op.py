#!/usr/bin/env python3
"""
Function creates the training operation of the network
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation

    Args:
    loss: loss of the network's prediction
    alpha: learning rate

    Returns:
    operation that trains the network using GD
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
