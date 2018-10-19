import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from curvefit import functions


def fit_and_plot(curve, name, plot, Xy):
    X = np.array(Xy[0])
    y = np.array(Xy[1])

    fitted_params, _ = curve_fit(curve, X, y, maxfev=500000)

    residuals = y - curve(X, *fitted_params)
    residual_sum_of_squares = np.sum(residuals ** 2)
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - (residual_sum_of_squares / total_sum_of_squares))

    smooth = np.linspace(-2, 2, 300)
    plot.plot(smooth, curve(smooth, *fitted_params), label=name + ": " + str(round(r_squared, 4)))


points = [
    [
        [-6.9, -4.8, -2.3, -1.2, -.3, .3, 1.3, 2.8, 4.7, 6.5],
        [-3.8, -3.5, -3.2, -2.1, -.5, .6, 1.9, 2.5, 2.9, 2.9]
    ],
    [
        [.8, 2.6, 4.6, 3.7, 5.7, 7.1, 9, 9.9],
        [-4.1, -1.2, 2.3, .7, .9, -1.2, -1, .4]
    ],
    [
        [1.2, 2.3, 3.6, 11.9, 12.2, 13.6, 14, 15],
        [-3.6, -3, -2.5, 4.6, 2.6, 2.3, 2, 1]
    ],
    [
        [.8, 2.6, 6.6, 7.1, 7.7, 8.3, 8.6, 8.8, 10.3, 12, 14.4, 16.3],
        [2.1, -1.2, -3.6, -2.9, -2.6, -2.7, -3.2, -3.7, -3.3, -1.8, -.8, -.5]
    ],
    [
        [1, 2.2, 4.8, 7.7, 9, 10.2, 11, 12],
        [2, -2.6, -4.4, -3.5, -2.1, .5, 1, 2]
    ],
    [
        [.4, 1, 1.6, 2.2, 2.5, 3.3, 9.8, 11.6, 5.6, 11.8, 12.1, 12.2, 12.6],
        [12.9, 6.3, 3.7, 1.5, .7, .38, -.4, .3, -.3, .7, 1.4, 2.4, 13.1]
    ]
]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled = []
for Xy in points:
    new_xy = []
    new_xy.append(scaler.fit_transform(Xy[0]))
    new_xy.append(scaler.fit_transform(Xy[1]))
    scaled.append(new_xy)

points = scaled
f, plots = plt.subplots(3, 2)
f.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

i = 0
for row in plots:
    for plot in row:
        Xy = points[i]
        i += 1

        plot.set_ylim([-2, 2])
        plot.scatter(Xy[0], Xy[1])

        for f in functions:
            fit_and_plot(f, f.__name__, plot, Xy)

        plot.legend()

plt.show()
