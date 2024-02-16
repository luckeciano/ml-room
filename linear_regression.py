from predictor import Predictor
from batch_sampler import BatchSampler
import numpy as np


class LinearRegression(Predictor):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.batch_sampler = None

    def train(self, train_x, train_y, nb_epochs = 1000, batch_size = 1000, lr = 0.1):
        self.weights = np.random.randn(*(1, train_x.shape[1]))
        self.bias = np.zeros((1, train_x.shape[1]))
        self.batch_sampler = BatchSampler(batch_size)

        for epoch in range(nb_epochs):
            batch_x, batch_y = self.batch_sampler.sample(train_x, train_y)
            y_hat = self.predict(batch_x)
            cost = self.__compute_cost(y_hat, batch_y)
            grad_w, grad_b = self.__compute_grad(batch_x, y_hat, batch_y)
            self.weights = self.weights - lr*grad_w
            self.bias = self.bias - lr*grad_b
            print("Epoch: " + str(epoch))
            print("Cost: " + str(cost))
            print("Gradients (W, b): " + str(grad_w)+ ", " + str(grad_b))
            print("Weights: " + str(self.weights) + ", " + str(self.bias))

    def predict(self, test_x):
        return self.weights * test_x + self.bias

    def __compute_cost(self, y_hat, y):
        return np.mean(np.fabs(y_hat**2 - y**2))
    
    def __compute_grad(self, batch_x, y_hat, train_y):
        grad_w = 2*np.mean((y_hat - train_y)*batch_x)
        grad_b = 2*np.mean(y_hat - train_y)
        return grad_w, grad_b
        
    

