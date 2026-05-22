#!/usr/bin/env python3
"""
Function returns two placeholders x and y for the neural network
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders x and y

    Args:
    nx: the number of feature columns in the data
    classes: the number of classes in the classifier

    Returns:
    x: placeholder for the input data
    y: placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
