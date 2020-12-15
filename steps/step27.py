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


x = Variable(np.array(np.pi/4))
y = sin(x)
y_ = my_sin(x)
y_.backward()
print(y.data - y_.data, x.grad)
x.name = 'x'
y_.name = 'y'
plot_dot_graph(y_, verbose=False, to_file='sin.png')
