import numpy as np


class PCA():    
    def compute_pca(self, X, dimensionality = None):
        if dimensionality is None:
            dimensionality = len(X)
        epsilon = 1e-3

        mean, std = np.mean(X, axis = 1, keepdims=True) , np.std(X, axis = 1, keepdims=True) + epsilon
        
        X = (X - mean)/std
        N = X.shape[1]
        S = (1.0/N)*X.dot(X.T)


        eigvals, eigvects = np.linalg.eig(S)

        eigvals = eigvals[0:dimensionality]
        eigvects = eigvects[:, 0:dimensionality]

        X_tilde = eigvects.dot(eigvects.T.dot(X)) * std + mean

        return eigvals, X_tilde


        
    

