# Author: Steven Yang 杨泰然 (steven_yang_0502 at outlook.com)

import numpy as np


def zero_centered(data_matrix):
    return data_matrix - np.mean(data_matrix, axis=0, keepdims=True)


def standardize(data_matrix):
    return (data_matrix - np.mean(data_matrix, axis=0, keepdims=True)) / np.std(data_matrix, axis=0, ddof=1)


def normalize(data_matrix):
    _range = np.max(data_matrix) - np.min(data_matrix)
    return (data_matrix - np.min(data_matrix)) / _range
