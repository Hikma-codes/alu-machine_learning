#!/usr/bin/env python3
'''
Script that defines a bidirectional RNN forward propagation function
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    '''
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: BidirectionalCell instance
        X: Input data (t, m, i)
        h_0: Initial hidden state for forward RNN (m, h)
        h_T: Initial hidden state for backward RNN (m, h)

    Returns:
        H: Concatenated hidden states (t, m, 2*h)
        Y: Outputs (t, m, o)
    '''
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Forward pass
    H_forward = np.zeros((t + 1, m, h))
    H_forward[0] = h_0
    for step in range(t):
        H_forward[step + 1] = bi_cell.forward(H_forward[step], X[step])

    # Backward pass
    H_backward = np.zeros((t + 1, m, h))
    H_backward[t] = h_T
    for step in range(t - 1, -1, -1):
        H_backward[step] = bi_cell.backward(H_backward[step + 1], X[step])

    # Concatenate hidden states
    H = np.concatenate((H_forward[1:], H_backward[:-1]), axis=-1)

    # Compute outputs
    Y = []
    for step in range(t):
        # Reshape to add batch dimension if needed
        h_concat = H[step][np.newaxis, :, :]  # Shape (1, m, 2*h)
        output = bi_cell.output(h_concat)
        Y.append(output[0])  # Remove batch dimension

    Y = np.array(Y)
    return H, Y
