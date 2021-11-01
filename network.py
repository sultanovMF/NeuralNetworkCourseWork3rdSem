import numpy as np

class Network:
    def __init__(self, topology):
        self._topology = topology

    def predict(self, input):
        output = input
        for layer in self._topology:
            output = layer.forward(output)

        return output

    def train(self, X, Y, epochs, learning_rate):
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.predict(x)
                # error
                error += self._topology[-1].calculate_loss(y, output)

                # backward
                grad = self._topology[-1].calculate_loss_diff(y, output)
                for layer in reversed(self._topology):
                    grad = layer.backward(grad, learning_rate)

            error /= len(X)
            print('%d/%d, error=%f' % (e + 1, epochs, error))

    def test(self, X, Y):
        errors = 0
        for x, y in zip(X, Y):
            output = self.predict(x)
            errors  += 1 if np.argmax(output) != np.argmax(y) else 0
            print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
        print(errors)