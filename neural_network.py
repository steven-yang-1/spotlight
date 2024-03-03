# Author: Steven Yang 杨泰然 (steven_yang_0502 at outlook.com)
# 无限隐藏层神经网络源码

import numpy
import math


def sigmoid(net):
    return 1.0 / (1.0 + numpy.exp(-net))


def d_sigmoid(x):
    return x * (1 - x)


class BPNeuralNetwork:
    Weights = []
    Weights_Last = None

    def __init__(self, hidden_nodes: list,
                 learning_rate=0.2,
                 converge_error_precision=0.8,
                 max_turns=100000,
                 debug=False,
                 name="General BPNetwork"):
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.converge_error_precision = converge_error_precision
        self.max_turns = max_turns
        self.debug = debug
        self.name = name

    def init_weights_matrix(self, x_len, y_len):
        # 生成元素的值为[0,1)的小随机数矩阵
        self.Weights.append(numpy.array(numpy.random.random((x_len, self.hidden_nodes[0]))))
        for i in range(len(self.hidden_nodes) - 1):
            self.Weights.append(numpy.array(numpy.random.random((self.hidden_nodes[i], self.hidden_nodes[i + 1]))))
        self.Weights_Last = numpy.array(numpy.random.random((self.hidden_nodes[len(self.hidden_nodes) - 1], y_len)))

    def fit(self, X_Dataset, Y_Dataset):
        self.init_weights_matrix(X_Dataset.shape[1], Y_Dataset.shape[1])
        turn = 0
        error = self.converge_error_precision + 1
        print("===== Model: " + self.name + " =====")
        while error > self.converge_error_precision and turn < self.max_turns:  # 收敛的条件
            error = 0.0
            turn = turn + 1
            for x_row_id, x_data in enumerate(X_Dataset):
                # 统计本轮中的误差
                y_data = Y_Dataset[x_row_id]

                hidden_nodes_values: list[list[float]] = []

                # 计算各隐藏结点的值和y_predict向量
                tmp = x_data
                for i in range(len(self.hidden_nodes)):
                    raw_nodes = numpy.matmul(numpy.array([tmp]), self.Weights[i])
                    raw_nodes = raw_nodes[0]
                    # 作sigmoid处理
                    nodes_values = []
                    for j in range(len(raw_nodes)):
                        nodes_values.append(sigmoid(raw_nodes[j]))
                    tmp = nodes_values
                    hidden_nodes_values.append(nodes_values)

                # 预测输出值
                _y_predict = numpy.matmul(numpy.array([tmp]), self.Weights_Last)
                if _y_predict.shape[0] == 1:
                    shape_coordinate = 1
                else:
                    shape_coordinate = 0

                y_predict = []
                # 作sigmoid处理
                for j in range(_y_predict.shape[shape_coordinate]):
                    y_predict.append(
                        sigmoid(_y_predict[0][j])
                    )

                # 累加error
                for i in range(len(y_data)):
                    error = error + math.pow(y_data[i] - y_predict[i], 2)

                # 计算输出层的梯度
                gradients_last_layer: list[float] = []
                for i in range(len(y_data)):
                    gradients_last_layer.append(
                        d_sigmoid(y_predict[i]) * (y_data[i] - y_predict[i])
                    )

                # 计算隐藏层的梯度
                gradients_layers: list[list[float]] = []
                for i in reversed(range(len(self.hidden_nodes))):
                    gradients_hidden = []
                    for j in range(self.hidden_nodes[i]):
                        z = 0.0
                        if i == len(self.hidden_nodes) - 1:
                            gradients = gradients_last_layer
                            Weights = self.Weights_Last
                        else:
                            gradients = gradients_layers[len(self.hidden_nodes) - i - 2]
                            Weights = self.Weights[i + 1]
                        for k in range(len(gradients)):
                            z = z + Weights[j, k] * gradients[k]
                        gradients_hidden.append(
                            z * d_sigmoid(hidden_nodes_values[i][j])
                        )
                    gradients_layers.append(gradients_hidden)

                # 1. 修改输出层的权重
                for k in range(len(self.hidden_nodes)):
                    for i in range(len(y_data)):
                        self.Weights_Last[k, i] = (self.Weights_Last[k, i] +
                                                   self.learning_rate *
                                                   hidden_nodes_values[len(self.hidden_nodes) - 1][k] *
                                                   gradients_last_layer[i])

                # 2. 修改隐藏层的权重
                for i in reversed(range(len(self.hidden_nodes))):
                    for j in range(self.hidden_nodes[i]):
                        if i == 0:
                            x = x_data
                            gradients = gradients_layers[len(gradients_layers) - 1]
                        else:
                            x = hidden_nodes_values[i - 1]
                            gradients = gradients_layers[len(gradients_layers) - i - 1]
                        for k in range(len(x)):
                            self.Weights[i][k, j] = (self.Weights[i][k, j] +
                                                     self.learning_rate *
                                                     (x[k] * gradients[j]))

            # 打印误差信息
            if self.debug:
                print("Training #" + str(turn) + " error: " + str(error))

    def predict(self, X_Dataset):
        result = []
        for row_index, x in enumerate(X_Dataset):
            result.append(self.predict_one(x))
        return result

    def predict_one(self, x_data):
        # 预测隐藏层
        tmp = x_data
        for i in range(len(self.hidden_nodes)):
            raw_nodes = numpy.matmul([tmp], self.Weights[i])
            raw_nodes = raw_nodes[0]
            # 作sigmoid处理
            nodes_values = []
            for j in range(len(raw_nodes)):
                nodes_values.append(sigmoid(raw_nodes[j]))
            tmp = nodes_values
        # 预测输出值
        last_raw_nodes = numpy.matmul([tmp], self.Weights_Last)
        if last_raw_nodes.shape[0] == 1:
            shape_coordinate = 1
        else:
            shape_coordinate = 0
        y_predict = []
        # 作sigmoid处理
        for j in range(last_raw_nodes.shape[shape_coordinate]):
            y_predict.append(
                sigmoid(last_raw_nodes[0][j])
            )
        return y_predict
