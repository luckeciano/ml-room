import numpy as np
idx = np.arange(20)
w = np.array([0.1, 0.2, -0.3, -0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -5.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10])

const = np.logical_or(np.greater(w, 0), np.greater(idx, 9)).astype(float)


print(const)