if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import platform
    if platform.system() == 'Darwin':
        sys.path.append(
            '/Users/youngin/anaconda3/envs/py-env/lib/python3.7/site-packages')

import numpy as np
from dezero import Variable, Parameter, Model
from dezero import optimizers
from dezero.utils import plot_dot_graph
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP

# 데이터 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 하이퍼파라미터 설정
lr = 0.2
max_iter = 10000
hidden_size = 10

# 모델 생성
model = MLP((16, 8, 4, 2, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)

# 학습
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    
    optimizer.update()
        
    if i % 1000 == 0:
        print(loss)