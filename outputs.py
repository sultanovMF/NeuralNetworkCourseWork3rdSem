import numpy as np
from output import Output


class TanhMSEOutput(Output):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_diff(x):
            return 1 - np.tanh(x) ** 2

        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_diff(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(tanh, tanh_diff, mse, mse_diff)


class SigmoidMSEOutput(Output):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_diff(x):
            s = sigmoid(x)
            return s * (1 - s)

        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_diff(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(sigmoid, sigmoid_diff, mse, mse_diff)
