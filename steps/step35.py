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


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

n = 5
for i in range(n-1):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    
# draw graph
gx = x.grad
gx.name = 'gx' + str(n)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')