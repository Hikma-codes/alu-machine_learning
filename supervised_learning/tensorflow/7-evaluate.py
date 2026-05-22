#!/usr/bin/env python3
"""
Function evaluates the output of the network
"""


import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of the network

    Args:
    X(numpy.ndarray): input data to evaluate
    Y(numpy.ndarray): one-hot labels for X
    save_path: location to load the model from

    Returns:
    network's prediction, accuracy and loss
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        ls = sess.run(loss, feed_dict={x: X, y: Y})
    return pred, acc, ls
