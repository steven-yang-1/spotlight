import numpy as np
import mixins


class MLR(mixins.DataPreprocessMixin):
    beta = []

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model(self):
        Xt = np.transpose(self.X)
        XtX = np.dot(Xt, self.X)
        Xty = np.dot(Xt, self.y)
        self.beta = np.linalg.solve(XtX, Xty)

    def predict(self, new_X):
        return np.dot(new_X, self.beta)

