import math
import numpy as np
import numpy.matlib
import mixins


class BPNeuralNetwork(mixins.DataPreprocessMixin):
    def __init__(self, X, Y, hidden_node_count=5, learn_rate=0.1, converge_precision=0.8, max_loop=10000):
        self.X = X
        self.Y = Y
        self.hidden_node_count = hidden_node_count
        self.learn_rate = learn_rate
        self.converge_precision = converge_precision
        self.max_loop = max_loop
        self.W = np.matlib.rand(self.hidden_node_count, Y.shape[1])
        self.V = np.matlib.rand(X.shape[1], self.hidden_node_count)
        self.activate_func = lambda net: BPNeuralNetwork.sigmoid(net)

    @staticmethod
    def sigmoid(net):
        return 1.0 / (1.0 + np.exp(-net))

    def model(self):
        n = 0
        eps = self.converge_precision + 1
        while eps > self.converge_precision and n < self.max_loop:
            eps = 0
            n = n + 1
            for row_index, x in enumerate(self.X):
                y = self.Y[row_index]
                XV = [x] * self.V
                OutputHidden = []
                for i in range(XV.shape[1]):
                    OutputHidden.append(self.activate_func(XV[0, i]))
                O1W = [OutputHidden] * self.W
                OutputVector = []
                for i in range(O1W.shape[0]):
                    OutputVector.append(self.activate_func(O1W[0, i]))
                Delta_O = []
                for i in range(len(y)):
                    delta_o = OutputVector[i]*(1 - OutputVector[i])*(y[i] - OutputVector[i])
                    Delta_O.append(delta_o)
                for i in range(len(y)):
                    eps = eps + math.pow(y[i] - OutputVector[i], 2)
                Delta_H = []
                for i in range(self.hidden_node_count):
                    z = 0
                    for j in range(len(y)):
                        z = z + self.W[i, j] * Delta_O[j]
                    delta_h = z * OutputHidden[i] * (1 - OutputHidden[i])
                    Delta_H.append(delta_h)
                for k in range(self.hidden_node_count):
                    for i in range(len(y)):
                        self.W[k, i] = self.W[k, i] + self.learn_rate * OutputHidden[k] * Delta_O[i]
                for k in range(len(x)):
                    for i in range(self.hidden_node_count):
                        self.V[k, i] = self.V[k, i] + self.learn_rate * x[k] * Delta_H[i]
            print(str(n) + "/Error: " + str(eps))

    def predict(self, X):
        Y = []
        for row_index, x in enumerate(X):
            XV = [x] * self.V
            OutputHidden = []
            for i in range(XV.shape[1]):
                OutputHidden.append(self.activate_func(XV[0, i]))
            O1W = [OutputHidden] * self.W
            OutputVector = []
            for i in range(O1W.shape[0]):
                OutputVector.append(self.activate_func(O1W[0, i]))
            Y.append(OutputVector)
        return Y
