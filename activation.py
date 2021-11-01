import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_diff):
        self._activation = activation
        self._activation_diff = activation_diff

    def forward(self, input):
        self.input = input
        return self._activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return self._calculate_gradient(output_gradient)

    def _calculate_gradient(self, output_gradient):
        return np.multiply(output_gradient, self._activation_diff(self.input))
