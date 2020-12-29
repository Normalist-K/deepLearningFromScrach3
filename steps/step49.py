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
from dezero import optimizers, datasets
import dezero.functions as F
from dezero.models import MLP


# set Hyperparameter
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# read data / create model, optimizer
train_set = datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 데이터셋의 인덱스 뒤섞기
    index = np.random.permutation(data_size)
    sum_loss = 0
    
    for i in range(max_iter):
        # 미니배치 생성
        batch_index = index[ i*batch_size : (i + 1)*batch_size ]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])
        
        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
        
    # 에포크마다 학습 결과 출력
    avg_loss = sum_loss / data_size
    print(f'epoch {epoch + 1}, loss {avg_loss:.2f}')
    