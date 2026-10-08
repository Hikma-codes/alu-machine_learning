#!/usr/bin/env python3
"""This module finds the best number of clusters using BIC for GMM"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters using the BIC"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    if type(kmin) is not int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None

    if kmin >= kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape

    best_bic = np.inf
    best_k = None
    best_result = None

    bics = []
    log_likelihoods = []

    for k in range(kmin, kmax + 1):
        res = expectation_maximization(X, k, iterations, tol, verbose)

        if res[0] is None:
            return None, None, None, None

        pi, m, S, g, ll = res

        # Calculate number of parameters
        # p = priors + means + covariance matrices
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Calculate BIC
        bic = p * np.log(n) - 2 * ll

        bics.append(bic)
        log_likelihoods.append(ll)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    bics = np.array(bics)
    log_likelihoods = np.array(log_likelihoods)

    return best_k, best_result, log_likelihoods, bics
