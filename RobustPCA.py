__author__ = 'artanis'

import numpy as np
from scipy.linalg import svd


def robust_pca(X, k, rand=np.random.RandomState(123)):
    """
    Robust principal components analysis

    :param X: n x d, treated as n d-dimensional elements
    :param k: number of components to keep
    :param rand: [RandomState(123)] random number generator
    :return:
           Y: n x k, X after dimensionality reduction
           P: d x k, each column is a principal component
           mu: d, mean of X
    """

    n, d = X.shape
    X = X.astype(np.float64, copy=False)
    eps = 1e-6

    if n == 1:
        U = np.zeros((d, k), dtype=np.float64)
        mu = X.flatten()
        return np.zeros((1, k), dtype=np.float64), U, mu
    else:
        mu = np.mean(X, axis=0)
        X -= mu

        # make sure X not too large or SVD slow O(min(d,n)^2.5)
        m = 2500
        if min(d, n) > m:
            X = X[rand.permutation(n)[:m]]
            n = m

        # get principal components using the SVD of X: X = U * S * V^T
        if d > n:
            U, S, _ = _robust_svd(np.dot(X, X.T) / (n - 1), rand=rand)
            s = [1.0 / np.sqrt(item) if abs(item) > eps else 0.0 for item in S]
            U = np.dot(np.dot(X.T, U), np.diag(s)) / np.sqrt(n - 1)
        else:
            U, S, _ = _robust_svd(np.dot(X.T, X) / (n - 1), rand=rand)

        # discard low variance principal components
        U = U[:, S > eps]
        U = U[:, :k]

        # perform dimensionality reduction
        P = np.zeros((d, k), dtype=np.float64)
        P[:, :U.shape[1]] = U
        return np.dot(X, P), P, mu


def _robust_svd(X, trials=100, rand=np.random.RandomState(123)):
    # Robust version of SVD more likely to always converge

    try:
        U, S, V = svd(X, full_matrices=False)
    except Exception as e:
        if trials <= 0:
            raise e
        else:
            size = X.size
            idx = rand.random_integers(low=0, high=size-1)
            X[idx / X.shape[1], idx % X.shape[1]] += np.spacing(1)
            U, S, V = _robust_svd(X, trials - 1, rand)

    return U, S, V

