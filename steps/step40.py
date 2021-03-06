if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import platform
    if platform.system() == 'Darwin':
        sys.path.append(
            '/Users/youngin/anaconda3/envs/py-env/lib/python3.7/site-packages')

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 / x1
print(y)

y.backward()
print(x1.grad)
