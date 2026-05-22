#!/usr/bin/env python3
"""
Function calculates the loss of a prediction
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the loss of a prediction

    Args:
    y: placeholder for the input data
    y_pred: placeholder of predictions

    Returns:
    loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
