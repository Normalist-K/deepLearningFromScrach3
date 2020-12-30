if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import platform
    if platform.system() == 'Darwin':
        sys.path.append(
            '/Users/youngin/anaconda3/envs/py-env/lib/python3.7/site-packages')

import math
import numpy as np
import dezero
from dezero import optimizers, datasets
from dezero import DataLoader, no_grad
import dezero.functions as F
from dezero.models import MLP


# set Hyperparameter
max_epoch = 3
batch_size = 100
hidden_size = 1000

# read data / create model, optimizer
train_set = datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0
    
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    print(f'epoch: {epoch+1}')
    print(f'train loss: {sum_loss/len(train_set):.4f}')

model.save_weight('my_mlp.npz')
    
    