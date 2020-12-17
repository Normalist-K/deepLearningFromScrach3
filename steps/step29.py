if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np

from dezero import Variable, Function
from dezero.utils import plot_dot_graph


def f(x):
    return x**4 - 2 * x**2


def gx2(x):
    return 12 * x**2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.clearGrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)

# lr = 0.001

# for i in range(iters):
#     print(i, x)

#     y = f(x)

#     x.clearGrad()
#     y.backward()

#     x_pre = x.data * 1.0
#     x.data -= x.grad * lr

#     if x_pre == x.data:
#         print(i, x)
#         break
