if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np

from dezero import Variable, Function
from dezero.utils import plot_dot_graph

def f(x):
    return x**4 - 2*x**2

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.clearGrad()
    y.backward(create_graph=True)
    
    gx = x.grad
    x.clearGrad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data