#!/usr/bin/env python3
"""
Function calculates the accuracy of a prediction
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    Args:
    y: placeholder for the labels of the input data
    y_pred: tenspr containing the network's predictions

    Returns:
    accuracy of the prediction
    """
    pred_max = tf.arg_max(y_pred, 1)
    y_max = tf.arg_max(y, 1)
    e = tf.equal(pred_max, y_max)
    return tf.reduce_mean(tf.cast(e, tf.float32))
