from sklearn.datasets import load_linnerud

X, y = load_linnerud(return_X_y=True)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from auto_curve import AutoCurver
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import numpy as np

scores = [[], [], [], [], [], [], [], [], [], [], []]

for _ in range(100):  # try many times, since our data is so small
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    for i, estimator in enumerate([
        MultiOutputRegressor(SVR()),
        MultiOutputRegressor(SVR(C=100, epsilon=1)),
        MultiOutputRegressor(GradientBoostingRegressor()),
        MultiOutputRegressor(GradientBoostingRegressor(loss="lad")),
        MultiOutputRegressor(RandomForestRegressor()),
        MultiOutputRegressor(RandomForestRegressor(criterion="mae")),
        MultiOutputRegressor(RandomForestRegressor(criterion="mae", max_features="sqrt")),
        MLPRegressor(),
        KernelRidge(),
        MultiOutputRegressor(AutoCurver(max_params=3)),
        MultiOutputRegressor(LinearRegression())
    ]):
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        error = np.mean(np.abs(predictions - y_test))
        scores[i].append(error)

print(list(np.mean(s) for s in scores))
