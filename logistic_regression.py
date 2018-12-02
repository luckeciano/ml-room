from predictor import Predictor
from batch_sampler import BatchSampler
import numpy as np
from metrics_tracker import MetricsTracker


class LogisticRegression(Predictor):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None
        self.batch_sampler = None
        self.metrics_tracker = MetricsTracker()

    def train(self, train_x, train_y, nb_epochs = 1000, batch_size = 1000, lr = 0.1, lambd = 0.000):
        self.weights = np.zeros(shape = (train_x.shape[0], 1))
        self.bias = 0
        self.batch_sampler = BatchSampler(batch_size)
        costs = []
        accuracies = []
        for epoch in range(nb_epochs):
            lr_decay = lr * (nb_epochs - epoch)/nb_epochs

            batch_x, batch_y = self.batch_sampler.sample(train_x, train_y)
            print("Shape batch_x: " + str(batch_x.shape))
            print("Shape train_y: " + str(batch_y.shape))

            y_hat = self.predict(batch_x)
            cost = self.__compute_cost(y_hat, batch_y, lambd)  
            grad_w, grad_b = self.__compute_grad(batch_x, y_hat, batch_y, lambd)
            self.weights = self.weights - lr_decay*grad_w
            self.bias = self.bias - lr_decay*grad_b

            if epoch%100 == 0:
                prediction = self.predict(train_x) > 0.5
                accuracy = self.metrics_tracker.accuracy(train_y, prediction)
                print("Epoch: " + str(epoch))
                print("Cost: " + str(cost))
                print("Gradients (W, b): " + str(grad_w)+ ", " + str(grad_b))
                print("Weights: " + str(self.weights) + ", " + str(self.bias))
                print("Accuracy: " + str(accuracy))

                costs.append(cost)
                accuracies.append(accuracy)
        return accuracies, costs

    def predict(self, X):
        a = np.dot(self.weights.T, X) + self.bias
        return self.__sigmoid(a)

    def __compute_cost(self, y_hat, y, lambd):
        eps = 10e-5
        return - np.mean(y * np.log(y_hat + eps) + (1-y) * np.log(1 - y_hat + eps)) + 0.5*lambd*np.sum(self.weights**2)
    
    def __compute_grad(self, X, y_hat, Y, lambd):
        m = X.shape[1]
        grad_w = (1/m) * np.dot(X, (y_hat - Y).T) + lambd * self.weights
        grad_b = (1/m) * np.sum(y_hat - Y)

        print("Shape grad_w: " + str(grad_w.shape))
        print("Shape grad_b: " + str(grad_b.shape))

        assert(grad_w.shape == self.weights.shape)
        return grad_w, grad_b

    def __sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
        
    

