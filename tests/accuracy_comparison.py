from sklearn.datasets import load_boston, load_diabetes, load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import numpy as np
from curvefit import SummedCurver, WeightedCurver
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split

import math


# lets start with regression


def score(estimator, new_X, new_y):
    estimator.fit(new_X, new_y)
    predictions = estimator.predict(X)
    error = np.mean((predictions - y) ** 2)

    if math.isnan(error):
        print(predictions)

    return -error


for l in [load_diabetes]:
    print("---------------------------------------------------" + str(l.__name__))
    for model in [WeightedCurver(max_params=3),
                  WeightedCurver(),
                  SVR(),
                  SummedCurver(max_params=3),
                  RandomForestRegressor(),
                  GradientBoostingRegressor(),
                  LinearRegression(),
                  HuberRegressor()]:
        print("testing " + model.__class__.__name__)
        X, y = l(return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)
        model.fit(train_X, train_y)
        print(score(model, test_X, test_y))

print("---------------------------------------------------load_linnerud")
X, y = load_linnerud(return_X_y=True)
for model in [
    MultiOutputRegressor(WeightedCurver(max_params=3)),
    MultiOutputRegressor(WeightedCurver()),
    MultiOutputRegressor(SummedCurver(max_params=3, method="dogbox")),
    MultiOutputRegressor(SVR()),
    MultiOutputRegressor(RandomForestRegressor()),
    MultiOutputRegressor(GradientBoostingRegressor()),
    MultiOutputRegressor(LinearRegression()),
    MultiOutputRegressor(HuberRegressor())]:
    print("testing " + model.estimator.__class__.__name__)
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    model.fit(train_X, train_y)
    print(score(model, test_X, test_y))
