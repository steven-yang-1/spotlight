# Author: Steven Yang 杨泰然 (steven_yang_0502 at outlook.com)
# 该文件包含了MLR、PCA和PCR算法的源代码
# MLR：多元线性回归
# PCA：主成分分析
# PCR：主成分回归

import numpy as np


class Algorithm:
    pass


class MLR(Algorithm):
    A = None

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def fit(self):
        Xt = np.transpose(self.X)
        XtX = np.dot(Xt, self.X)
        Xty = np.dot(Xt, self.y)
        self.A = np.linalg.solve(XtX, Xty)

    def predict(self, X):
        return np.dot(X, self.A)


class PCA(Algorithm):
    T = None
    P = None
    snr_compare = None

    def __init__(self, X):
        self.X = X
        self.svd_decompose()

    def svd_decompose(self):
        U, sigma, V = np.linalg.svd(self.X, full_matrices=False)
        i = len(sigma)
        S = np.zeros((i, i))
        S[:i, :i] = np.diag(sigma)
        self.T = U.dot(U, S)
        V = V.T
        self.P = V
        snr_compare = []
        for i in range(len(sigma) - 1):
            temp = sigma[i] / sigma[i + 1]
            snr_compare.append(temp)
        self.snr_compare = snr_compare
        return U, S, V, snr_compare

    def pca_decompose(self, components):
        T = self.T[:, :components]
        P = self.P[:, :components]
        return T, P


class PCR(Algorithm):
    A = None

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def fit(self, components):
        pca = PCA(self.X)
        T, P = pca.pca_decompose(components)
        mlr = MLR(T, self.Y)
        mlr.fit()
        self.A = np.dot(P, mlr.A)

    def predict(self, X):
        Y = np.dot(X, self.A)
        return Y