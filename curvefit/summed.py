from sklearn.base import BaseEstimator, RegressorMixin
from inspect import signature
import numpy as np
from .functions import function_lists
from .helpers import chunks, safe_curve_fit, random_search_curve_fit, r_squared


class SingleExtender(object):
    """Extends 1 math function to work over multidimensional X"""

    # no longer used but makes mulitextender easier to understand

    def __init__(self, function):
        self.function = function
        self.n_vars = len(signature(function).parameters) - 1

    def call(self, X, *opts):
        opts = list(opts)

        # split opts into function-opt sized chunks
        opts = list(chunks(opts, self.n_vars))

        return np.sum([self.function(X[i], *opt) for i, opt in enumerate(opts)])


class MultiExtender(object):
    """extends n math functions to work over n dimensions"""

    def __init__(self, functions):
        self.functions = functions
        self.n_vars_per_f = [len(signature(function).parameters) - 1 for function in functions]
        self.total_vars = np.sum(self.n_vars_per_f)

    def call(self, X, *opts):
        opts = list(opts)
        total = 0
        index = 0
        for i, f in enumerate(self.functions):
            total += f(X[i], *opts[index: index + self.n_vars_per_f[i]])
            index += self.n_vars_per_f[i]
        return total


class SummedCurver(BaseEstimator, RegressorMixin):
    """
    fit a function to each dimension, and when prediction return the sum of each functions prediction for its dimension
    """

    def __init__(self, max_params=8, maxfev=3000, function_type="common", method="lm"):
        self.max_params = max_params
        self.function_type = function_type
        functions = function_lists[function_type]

        # filter functions according to type and max_params
        self.estimators = []
        self.functions = []
        for f in functions:
            if len(signature(f).parameters) <= max_params:
                self.functions.append(f)

        # params for scipy curve_fit
        self.maxfev = maxfev
        self.method = method

    def fit(self, X, y):
        rot = np.rot90(X)  # we rotate so we can loop over dimensions instead of samples

        best_functions = []
        for dimension in rot:
            best_score = None
            best_fun = None
            for fun in self.functions:
                params = safe_curve_fit(fun, dimension, y, len(signature(fun).parameters) - 1, maxfev=self.maxfev,
                                        method=self.method)
                s = r_squared(params, fun, dimension, y)
                if best_score is None or s > best_score:  # we save the best function to represent each dimension
                    best_score = s
                    best_fun = fun
            best_functions.append(best_fun)

        m = MultiExtender(best_functions)

        params = random_search_curve_fit(m.call, rot, y, m.total_vars,
                                         maxfev=self.maxfev * 5,  # more variables -> higher maxfev
                                         method=self.method)

        self.m = m
        self.params = params

        return self

    def predict(self, X):
        return np.array(self.m.call(np.rot90(X), *self.params))
