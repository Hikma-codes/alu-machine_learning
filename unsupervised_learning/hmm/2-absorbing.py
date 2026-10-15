#!/usr/bin/env python3
"""
Function that determines if a Markov chain is absorbing
"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]
    absorbing_states = np.isclose(np.diag(P), 1)

    if not np.any(absorbing_states):
        return False

    # Create reachability matrix using powers of P
    reachable = P.copy()
    power = P.copy()
    for _ in range(n):
        power = np.matmul(power, P)
        reachable += power

    # Check if each state can reach at least one absorbing state
    for i in range(n):
        if not np.any(reachable[i][absorbing_states] > 0):
            return False

    return True
