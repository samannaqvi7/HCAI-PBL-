import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def neg_log_posterior(w, rankings, reg):
    """rankings: list of (n_k, d) arrays, rows in preference order (best first).
    A length-2 ranking is a plain pairwise comparison, so this one function is the
    Plackett-Luce extension of Bradley-Terry: at n=2 it is exactly the Bradley-Terry
    negative log-likelihood (see tests in verify script)."""
    nll = 0.0
    for ranking in rankings:
        utilities = ranking @ w  # (n_k,), best-first order
        for k in range(len(utilities) - 1):
            remaining = utilities[k:]
            nll += logsumexp(remaining) - utilities[k]
    return nll + 0.5 * reg * np.dot(w, w)


def fit_preference_vector(rankings, n_features, reg=1.0):
    """MAP estimate of w under a Gaussian(0, 1/reg) prior. rankings must be non-empty."""
    w0 = np.zeros(n_features)
    result = minimize(neg_log_posterior, w0, args=(rankings, reg), method='L-BFGS-B')
    return result.x


def predict_utility(w, X):
    return X @ w


def bradley_terry_neg_log_likelihood(w, x_winner, x_loser):
    """Closed-form pairwise Bradley-Terry negative log-likelihood, used only to verify
    that neg_log_posterior's n=2 case matches it exactly (see verify script)."""
    diff = np.dot(w, x_winner - x_loser)
    return -np.log(1.0 / (1.0 + np.exp(-diff)))
