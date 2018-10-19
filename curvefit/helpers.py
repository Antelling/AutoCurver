from scipy.optimize import curve_fit
import numpy as np
import math


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def r_squared(params, curve, X, y):
    # computes r_score
    residuals = y - curve(X, *params)
    residual_sum_of_squares = np.sum(residuals ** 2)
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_square = float(1 - (residual_sum_of_squares / total_sum_of_squares))
    return r_square

def contain_nan(params):
    for p in params:
        if math.isnan(p):
            return True
    return False


def random_search_curve_fit(f, X, y, n_opts, maxfev=3000, method="lm", attempts=20):
    best_score = None
    best_params = []
    for attempt in range(attempts):
        params = safe_curve_fit(f, X, y, n_opts, maxfev=maxfev, method=method)
        s = r_squared(params, f, X, y)
        if best_score is None or s > best_score:  # we save the best function to represent each dimension
            best_score = s
            best_params = params
    return best_params


def safe_curve_fit(f, X, y, n_opts, maxfev=3000, method="lm"):
    """fit params to a function, guaranteed to succeed.
    f is the function
    X and y are input and labels
    n_opts is the number of arguments f accepts
    maxfev and method are parameters for scipy curve_fit"""
    fails = 1

    while True:
        try:
            p0 = np.random.uniform(-1 * fails, 1 * fails, n_opts)  # gradually increase the magnitude of our guess
            params, _ = curve_fit(f, X, y, p0=p0, maxfev=maxfev, method=method)
            assert not contain_nan(params)
            break
        except (RuntimeError, ValueError, AssertionError):
            fails += 1
            if fails > 30:
                params = np.random.uniform(-1, 1, n_opts)  # screw it, make up some numbers
                break
    return params
