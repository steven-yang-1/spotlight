# Author: Steven Yang 杨泰然 (steven_yang_0502 at outlook.com)

import neural_network
import numpy

# 这是神经网络的测试文件
# 实验证明到了5层以内分别10个结点的异或函数都是很快收敛的，
# 有兴趣的筒子们可以一起参与测试和Debug，谢谢！

bp_network = neural_network.BPNeuralNetwork([10, 10, 10, 10, 10],   # 这个数组代表建立5个分别为10个隐藏结点的隐藏层
                                            learning_rate=4.2,  # 学习率
                                            converge_error_precision=0.08,  # 收敛精度
                                            debug=True) # 是否输出误差信息

X_Dataset = numpy.array([
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

Y_Dataset = numpy.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0]
])

# 建模
bp_network.fit(X_Dataset, Y_Dataset)

Y_Dataset_Predict = bp_network.predict(X_Dataset)

print(Y_Dataset_Predict)
