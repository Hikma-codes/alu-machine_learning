#!/usr/bin/env python3

"""
This module contains a function that performs
expectation maximization for a GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k,
                             iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer for number of clusters
    - iterations: max number of iterations for EM algorithm
    - tol: tolerance for log likelihood convergence
    - verbose: if True, prints log likelihood every 10 iterations

    Returns:
    - pi: numpy.ndarray of shape (k,) with priors for each cluster
    - m: numpy.ndarray of shape (k, d) with centroid means
    - S: numpy.ndarray of shape (k, d, d) with covariance matrices
    - g: numpy.ndarray of shape (k, n) with posterior probabilities
    - log_likelihood: the final log likelihood of the model
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, log_likelihood = expectation(X, pi, m, S)
    prev_like = log_likelihood
    msg = "Log Likelihood after {} iterations: {}"

    if verbose:
        print(msg.format(0, round(log_likelihood, 5)))

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, log_likelihood = expectation(X, pi, m, S)

        if verbose and (i + 1) % 10 == 0:
            print(msg.format(i + 1, round(log_likelihood, 5)))

        if abs(prev_like - log_likelihood) <= tol:
            break

        prev_like = log_likelihood

    if verbose and (i + 1) % 10 != 0:
        print(msg.format(i + 1, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
