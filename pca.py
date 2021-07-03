import numpy as np


class PCA:
    def __init__(self, X, components=3):
        """
        X: 数据矩阵，每一行是一个样本
        components: 要降到的维数，应小于X的列数
        """
        self.X = X
        self.components = components
        self.eigenvalues = []
        self.eigenvectors = []

    def analyze(self):
        sigma = np.cov(self.X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        result_set = []
        for i, x in enumerate(eigenvalues):
            result_set.append((x, eigenvectors[:, i]))
        result_set = sorted(result_set, key=lambda th: th[0], reverse=True)
        self.eigenvalues = [x[0] for i, x in enumerate(result_set)]
        self.eigenvectors = np.array([x[1] for i, x in enumerate(result_set)]).T
        Y = np.dot(self.X, self.eigenvectors[:, :self.components])
        return Y
