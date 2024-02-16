from predictor import Predictor
import numpy as np


class BayesianLinearRegression(Predictor):
    def __init__(self):
        self.m_n = None
        self.S_n = None
        self.sigma = None

    def train(self, train_x, train_y, sigma, m0, s0):
        bias = np.ones((train_x.shape[0], 1))
        X = np.concatenate((train_x, bias), axis = 1)
        inv_s0 = np.linalg.inv(s0)
        self.S_n = np.linalg.inv(inv_s0 + (1/(sigma**2))*X.T.dot(X))
        self.sigma = sigma        
        self.m_n = self.S_n.dot(inv_s0.dot(m0) + (1/(sigma**2))*X.T.dot(train_y))
  

    def predict(self, test_x):
        bias = np.ones((test_x.shape[0], 1))
        X = np.concatenate((test_x, bias), axis = 1)
        mu = X.dot(self.m_n).flatten()
        var = np.array(map(lambda x: x.dot(self.S_n).dot(x.T)+ self.sigma**2, X))
        return np.random.normal(mu, var), mu, var
        
    

