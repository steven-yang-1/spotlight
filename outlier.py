import numpy as np
import numpy.linalg


class MahalanobisDistance:
    def __init__(self, X):
        self.X = X
        cov_mat = np.cov(self.X.T)
        self.inv_cov_of_X = np.linalg.inv(cov_mat)
        self.expect_of_X = np.mean(X, axis=0, keepdims=True)

    def compute(self, x):
        return np.dot(np.dot((x - self.expect_of_X), self.inv_cov_of_X), np.transpose(x - self.expect_of_X))[0][0]
