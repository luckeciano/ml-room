import numpy as np
import time

class MetricsTracker():

    def mean_squared_error(self, y_hat, y):
        return np.mean(np.fabs(y_hat**2 - y**2))

    def mean_absolute_error(self, y_hat, y):
        return np.mean(np.fabs(y_hat - y))

    def profile(self, operation, *args):
        start = time.time()
        res = operation(*args)
        end = time.time()
        print (operation.__name__ + ": " + str(end - start) + " seconds.")
        return res

    def accuracy(self, y, y_hat):
        assert(y.shape == y_hat.shape)
        return np.sum(y == y_hat)/y.shape[1]