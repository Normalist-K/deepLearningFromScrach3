if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np

from dezero import Variable, Function
from dezero.utils import plot_dot_graph


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return gy * np.cos(x)


def sin(x):
    return Sin()(x)


def my_sin(x, threashold=0.0001):
    y = 0
    for i in range(1000000):
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x ** (2*i + 1)
        y = y + t
        if abs(t.data) < threashold:
            break
    return y


def rosenbrock(x0, x1):
    return 100 * (x1 - x0**2)**2 + (1 - x0)**2


x0 = Variable(np.array(0.99))
x1 = Variable(np.array(1.0))
lr = 0.001
iters = 1000

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.clearGrad()
    x1.clearGrad()
    y.backward()

    x0.data -= x0.grad * lr
    x1.data -= x1.grad * lr
