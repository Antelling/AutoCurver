from sklearn.base import BaseEstimator
import numpy as np
import inspect
from .helpers import r_squared, safe_curve_fit
from .functions import function_lists


class WeightedCurver(BaseEstimator):
    """
    fit a function to each dimension, and when prediction return the r_squared weighted average of each functions
    prediction for its dimension
    """
    def __init__(self, max_params=8, maxfev=3000, certainty_scaler=1, function_type="common", method="lm"):
        self.max_params = max_params
        self.function_type = function_type
        functions = function_lists[function_type]

        #filter functions according to type and max_params
        self.estimators = []
        self.functions = []
        for f in functions:
            if len(inspect.signature(f).parameters) <= max_params:
                self.functions.append(f)

        #params for scipy curve_fit
        self.maxfev = maxfev
        self.method = method
        self.certainty_scaler = certainty_scaler

    def fit(self, X, y):
        # produce a new estimator for each dimension
        # an estimator is a dictionary of the function, its params, and its rscore
        dimensions = X.shape[1]
        for dimension in range(dimensions):
            x = [x[dimension] for x in X]
            self.estimators.append(self._fit_on_one_dimension(x, y))
        return self

    def _fit_on_one_dimension(self, x, y):
        np.seterr(
            all="ignore")  # numpy likes to throw an error if you like give log a negative value. We don't want that
        best = {"params": [], "score": 0, "f": {}}
        for f in self.functions:
            score, params = self._fit_func(f, x, y)
            if score > best["score"]:
                best = {"params": params, "score": score, "f": f}
        return best

    def _fit_func(self, f, x, y):
        x = np.array(x)
        y = np.array(y)

        n_opts = len(inspect.signature(f).parameters) - 1 #-1 because its f(x, a, b, c...) and we don't care about x
        # fit the function, then return rsquared and params
        fitted_params = safe_curve_fit(f, x, y, n_opts, maxfev=self.maxfev, method=self.method)
        r_square = r_squared(fitted_params, f, x, y)

        return r_square, fitted_params

    def _weighted_average(self, point):
        #point is a list of value, r_score tuples
        #return the average value using r_score as the weight
        total = np.sum([x[1] for x in point])
        weights = [x[1] / total for x in point]
        data = [x[0] for x in point]
        return np.average(data, weights=weights)

    def predict(self, X):
        # for every point, get a prediction and r2, then compute the weighted
        Y = []
        for point in X:
            new_point = []
            for i, dimension in enumerate(point):
                e = self.estimators[i]
                y = e["f"](np.array([dimension]), *e["params"])[0]
                new_point.append((y, e["score"] ** self.certainty_scaler))
            Y.append(self._weighted_average(new_point))
        return np.array(Y)
