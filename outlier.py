import numpy as np
import numpy.linalg


class MahalanobisDistance:
    def __init__(self, X):
        self.X = X
        cov_mat = np.cov(self.X.T)
        assert np.linalg.det(cov_mat) != 0.0
        self.inv_cov_of_X = np.linalg.inv(cov_mat)
        self.expect_of_X = np.mean(X, axis=0, keepdims=True)

    def __call__(self, x, y=None):
        if y is None:
            y = self.expect_of_X
        return np.sqrt(np.dot(np.dot((x - y), self.inv_cov_of_X), np.transpose(x - y))[0][0])