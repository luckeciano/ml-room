import numpy as np

class BatchSampler():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self, train_x, train_y):
        shuffle_idx = np.arange(train_x.shape[1])
        np.random.shuffle(shuffle_idx)
        batch_idx = shuffle_idx[:self.batch_size]
        return (np.take(train_x, batch_idx, axis=1), np.take(train_y, batch_idx, axis=1))
