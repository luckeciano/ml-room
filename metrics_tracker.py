import numpy as np
import time

class MetricsTracker():

    def mean_squared_error(y_hat, y):
        return np.mean(np.fabs(y_hat**2 - y**2))

    def mean_absolute_error(y_hat, y):
        return np.mean(np.fabs(y_hat - y))

    def profile(operation, **args):
        start = time.time()
        operation(**args)
        end = time.time()
        print (operation.__name__ + ": " + str(end - start))
        return end - start

    def accuracy(self, y, y_hat):
        assert(y.shape == y_hat.shape)
        return np.sum(y == y_hat)/y.shape[1]