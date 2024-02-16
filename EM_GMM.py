from predictor import Predictor
from scipy.stats import multivariate_normal
import numpy as np


class EM_GMM():
    def compute_EM(self, X, k, steps = 1000):
        n = X.shape[1] # number of data points

        #means = [np.ones(n)] * k
        means = [-4, 0, -8]
        covariances = [1, 0.2, 3]
        #covariances = [np.eye(n)] * k
        
        weights = np.ones(k) / k

        for _ in range(steps):
            r = self.__evaluation(X, means, covariances, weights)

            means, covariances, weights = self.__maximization(X, means, covariances, r)
        print(means, covariances, weights)
        return means, covariances, weights, r

    def __evaluation(self, X, means, covariances, weights):
        k = len(means)
        samples = np.array([multivariate_normal.pdf(X, means[i], covariances[i]) for i in range(k)]).T
        unnormalized_r = weights.T * samples
        norm_factor = unnormalized_r.sum(axis = 1, keepdims=True)
        r = unnormalized_r/norm_factor
        return r
    
    def __maximization(self, X, means, covariances, r):
        k = len(means)
        Nk = np.sum(r, axis = 0)
        means = (1/Nk) * np.sum(r * X, axis = 0)
        for i in range(k): 
            c = np.array(map(lambda x: ( x - means[i] ).dot( (x-means[i]).T), X))
            covariances[i] = np.array((1/Nk[i]) * np.sum( r[:,i] * c) )
        # covariances = np.array([ ( (1/Nk) * np.sum( r.T.dot(
        #     np.array(
        #      )  ), axis = 1 ) ) for i in range(k)])
        #print(covariances)
        weights =  Nk/np.sum(Nk)
    
        return means, covariances, weights
        





        
    

