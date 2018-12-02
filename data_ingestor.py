import pandas as pd
import numpy as np

class DataIngestor():
    def read_csv(self, filepath):
        df = pd.read_csv(filepath, sep = ',', dtype=float)
        X = np.array(df.values[:, :-1])
        Y = np.array(df.values[:, -1:])
        return X.T, Y.T  #last column is label

        