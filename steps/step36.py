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

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph = True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)