from layers import FullyConnected, Tanh, Softmax, ReLU
from network import Loss, Network
from keras.datasets import mnist
from keras.utils import set_random_seed
from utils import (one_hot_encode, cross_entropy_loss, cross_entropy_loss_derivative,
                   squared_error, squared_error_derivative)
import matplotlib.pyplot as plt


if __name__ == "__main__":
    set_random_seed(123)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = one_hot_encode(y_train)

    layers = [FullyConnected(input_size=784, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=10), Tanh()]

    layers2 = [FullyConnected(input_size=784, output_size=16), Tanh(),
               FullyConnected(input_size=16, output_size=16), Tanh(),
               FullyConnected(input_size=16, output_size=10), Tanh(),
               Softmax()]

    layers3 = [FullyConnected(input_size=784, output_size=512), ReLU(),
               FullyConnected(input_size=512, output_size=10), Tanh(),
               Softmax()]

    loss = Loss(squared_error, squared_error_derivative)
    loss2 = Loss(cross_entropy_loss, cross_entropy_loss_derivative)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape(((10000, 28 * 28)))
    x_test = x_test.astype('float32') / 255

    net = Network(layers3, learning_rate=0.1)
    net.compile(loss2)

    history = net.fit(x_train, y_train, 5, verbose=1, batch_size=128, learning_rate=0.3)
    accuracy = net.evaluate(x_test, one_hot_encode(y_test))

    print(f"Accuracy: {accuracy:.2f}%")
    plt.plot(history.keys(), history.values())
    plt.show()

