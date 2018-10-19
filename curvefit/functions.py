import numpy as np
import math


def sign(x):
    return -1 if x < 0 else 1


# region poly

def linear(x, a, b):
    return a * x + b


def quartic(x, a, b, c):
    return a * x ** 2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 * c * x + d


def quadratic(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def a_to_x(x, a, b, c):
    return a ** (x + b) + c


def sqrt(X, a, b, c):
    return [sign(x) * a * np.sqrt(np.abs(x)) + b * np.abs(x) + c for x in X]


def cbrt(X, a, b, c, d):
    return [sign(x) * a * np.cbrt(np.abs(x)) + b * np.sqrt(np.abs(x)) + c * np.abs(x) + d for x in X]


# endregion
# region distributions
def normal(x, center, spread, slide_ver, stretch_ver):
    left = 1 / np.sqrt(2 * math.pi * np.abs(spread))
    right = - ((x - center) ** 2) / (2 * spread)
    return slide_ver + stretch_ver * (left * np.exp(right))


def seminormal(x, center, spread1, spread2, slide_ver, stretch_ver):
    left = 1 / np.sqrt(np.abs(spread1))
    right = ((x - center) ** 2) / (spread2)
    return slide_ver + stretch_ver * left * np.exp(right)


def laplace(x, u, b, ver_slide, ver_stretch):
    y = 1 / (2 * b) * np.exp(-np.abs(x - u) / b)
    return y * ver_stretch + ver_slide


def cauchy(x, u, l, ver_slide, ver_stretch):
    y = (1 / (math.pi * l)) * (l ** 2 / ((x - u) ** 2 + l ** 2))
    return y * ver_stretch + ver_slide


# endregion
# region periodic
def sinusoidal(x, slide_ver, slide_hor, stretch_hor, stretch_ver):
    internal = x * stretch_hor + slide_hor
    return slide_ver + stretch_ver * np.sin(internal)


def sin_plus_normal(x, a, b, c, d, u, s, e, f, l):
    return sinusoidal(x, a, b, c, d) + seminormal(x, u, s, e, f, l)


def sin_times_x(x, a, b, c, d, u, s):
    return sinusoidal(x, a, b, c, d) * linear(x, u, s)


def sin_plus_x(x, a, b, c, d, u, s):
    return sinusoidal(x, a, b, c, d) + linear(x, u, s)


# endregion
# region other
def logistic(x, max, steep, slide_hor, slide_ver):
    steep /= 3  # we want to slow down how fast scipy goes over this one, or it sometimes errors out
    return max / (1 + np.exp(-steep * (x + slide_hor))) + slide_ver


def logistic2(x, max, steep, slide_hor, slide_ver):
    steep += 1  # even with slowing, it sometimes flies to infinity, so we have this one with slightly different params
    return -max / (1 + np.exp(-steep * (x + slide_hor))) + slide_ver


def logarithmic(x, slide_ver, stretch_ver, stretch_hor, slide_hor):
    return slide_ver + stretch_ver * np.log(np.abs((x + slide_hor) * stretch_hor))


function_lists = {
    "all": [linear, quartic, cubic, quadratic, a_to_x, sqrt, cbrt, normal, seminormal, laplace, cauchy, sinusoidal,
            sin_plus_normal, sin_times_x, sin_plus_x, logistic, logistic2],
    "poly": [linear, quartic, cubic],
    "common": [linear, quartic, cubic, logistic, logarithmic, seminormal]
}
# endregion
