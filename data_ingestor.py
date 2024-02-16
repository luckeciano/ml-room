import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataIngestor():
    def read_csv(self, filepath):
        df = pd.read_csv(filepath, sep = ',', dtype=float)
        X = np.array(df.values[:, :-1])
        Y = np.array(df.values[:, -1:])
        return X.T, Y.T  #last column is label

    def load_mnist(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

        X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
        y_train = mnist.train.labels

        X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
        y_test = mnist.test.labels

        del mnist

        return X_train, y_train, X_test, y_test
        