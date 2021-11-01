from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from activations import Tanh
from network import Network
from outputs import TanhMSEOutput

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)

x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

topology = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    TanhMSEOutput()
]

epochs = 1000
learning_rate = 0.05

network = Network(topology)
network.train(x_train[:1000], y_train[:1000], epochs, learning_rate)
network.test(x_test[0:100], y_test[0:100])