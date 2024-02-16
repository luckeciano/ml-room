from predictor import Predictor
from batch_sampler import BatchSampler
import numpy as np
from metrics_tracker import MetricsTracker


class MLP(Predictor):
    def __init__(self, n_x, n_hidden_layers, neurons_per_layer, n_output, activations):
        super().__init__()
        self.initialize_parameters(n_x, n_hidden_layers, neurons_per_layer, n_output)

        self.batch_sampler = None
        self.activations = activations
        self.act_dict = {"sigmoid": self.__sigmoid, "relu": self.__relu, "linear": self.__linear}
        self.metrics_tracker = MetricsTracker()
        self.n_layers = n_hidden_layers + 2

        assert(len(activations) == n_hidden_layers + 1)

    def train(self, train_x, train_y, nb_epochs = 1000, batch_size = 1000, lr = 0.1):
        self.batch_sampler = BatchSampler(batch_size)
        costs = []
        accuracies = []

        for epoch in range(nb_epochs):
            lr_decay = lr * (nb_epochs - epoch)/nb_epochs

            batch_x, batch_y = self.batch_sampler.sample(train_x, train_y)
            print("Shape batch_x: " + str(batch_x.shape))
            print("Shape train_y: " + str(batch_y.shape))

            cost, y_hat = self.__forward_prop(batch_x, batch_y)

            grad = self.__backward_prop(batch_x, batch_y, y_hat)

            self.learning_update(grad, lr)

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
    
    def initialize_parameters(self, n_x, n_hidden_layers, neurons_per_layer, n_output):
        assert(len(neurons_per_layer) == n_hidden_layers + 1)
        W0 = np.random.rand(shape = (neurons_per_layer[0], n_x))
        b0 = np.zeros(shape = (neurons_per_layer[0], 1))
        self.weights = {"W0": W0, "b0": b0}
        
        for i in range(n_hidden_layers):
            Wi = np.random.rand(shape = (neurons_per_layer[i + 1], neurons_per_layer[i]))
            bi = np.zeros(shape = ((neurons_per_layer[i + 1], 1)))
            self.weights["W" + str(i+1)] = Wi
            self.weights["b" + str(i+1)] = bi
        
        W_output = np.random.rand(shape = (n_output, neurons_per_layer[n_hidden_layers - 1]))
        b_output = np.zeros(shape = ((n_output, 1)))
        self.weights["W" + str(n_hidden_layers+1)] = W_output
        self.weights["b" + str(n_hidden_layers+1)] = b_output

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __relu(self, x):
        return np.max(0, x, axis=1)

    def __linear(self, x):
        return x

    def __forward_prop(batch_x, batch_y):
        A_last = batch_x
        for i in range(self.n_layers):
            Wi = self.weights["W" + str(i)]
            bi = self.weights["b" + str(i)]
            Z = Wi * A_last + bi
            A_output = self.act_dict[self.activations[i]]
            A_last = A_output
        m = batch_x.shape[1]
        eps = 10e-5
        cost = (-1/m) * np.sum(Y * np.log(A_output + eps) + (1 - Y)*np.log(1 - A_output + eps))

        assert(cost.shape == ())
        assert(A_output.shape == (1, m))

        cost = np.squeeze(cost) 

        return cost, A_output

    def __backward_prop(batch_x, batch_y, y_hat):


    

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
        
    

