import network
import numpy as np
import data_io
import outlier
import pca


def bp_network():
    samples_X = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1]
    ])
    samples_Y = np.array([
        [1],
        [1],
        [0],
        [0]
    ])
    bp = network.BPNeuralNetwork(
        samples_X,
        samples_Y,
        hidden_node_count=6,
        learn_rate=0.5,
        converge_precision=0.01
    )
    bp.model()

    #data_io.DataIO.export_to(bp, "D:\\tmp.model")

    #bp = data_io.DataIO.load("D:\\tmp.model")

    #print(bp)

    predict_result = bp.predict(np.array([
        [0, 1],
        [0, 0]
    ]))
    print(predict_result)
    # correct result: 1, 0


def pca_algorithm():
    pca_object = pca.PCA([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ], components=1)
    print(pca_object.analyze())
    print(pca_object.eigenvalues)
    print(pca_object.eigenvectors)


def pca_by_svd():
    pca_object = pca.PCA([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ], components=1)
    print(pca_object.analyze_by_svd())
    print(pca_object.eigenvalues)
    print(pca_object.eigenvectors)


def m_dist():
    data = np.array([
        [1.0, 2.0, 1.0],
        [2.1, 1.0, 2.5],
        [3.5, 1.0, 3.2],
        [2.0, 2.0, 4.3]
    ])
    m_distance = outlier.MahalanobisDistance(data)
    print(m_distance.compute([1, 2, 3]))


if __name__ == '__main__':
    #pca_by_svd()
    #bp_network()
    m_dist()
