#!/usr/bin/env python3
"""
Function creates a forward propagation graph
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates a forward propagation graph

    Args:
    x: placeholder
    layer_sizes: number of nodes in each layer list
    activations: activation functions list for each layer

    Returns:
    prediction of the network
    """
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
