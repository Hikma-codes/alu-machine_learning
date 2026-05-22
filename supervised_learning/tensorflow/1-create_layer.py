#!/usr/bin/env python3
"""
Function creates a layer
"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a new layer

    Args:
    prev: tensor output of the previous layer
    n: number of nodes in the layer to create
    activation: activation function

    Returns:
    prev: tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init)
    return layer(prev)
