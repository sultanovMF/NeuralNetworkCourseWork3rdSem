import numpy as np

from dense import Dense
from activations import Tanh
from outputs import TanhMSEOutput
from network import Network

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

epochs = 10000
learning_rate = 0.1

topology = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    TanhMSEOutput()
]

network = Network(topology)
network.train(X, Y, epochs, learning_rate)

for x in X:
    print(x.tolist(), '->', network.predict(x))