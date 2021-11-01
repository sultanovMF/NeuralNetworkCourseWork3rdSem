import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_diff):
        self.activation = activation
        self.activation_prime = activation_diff

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return self._calculate_gradient(output_gradient)

    def _calculate_gradient(self, output_gradient):
        return np.multiply(output_gradient, self.activation_prime(self.input))
