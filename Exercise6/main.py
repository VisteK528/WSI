from layers import FullyConnected, Tanh
from network import Loss, Network
import numpy as np
"""from keras.datasets import mnist
from keras.utils import set_random_seed"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def cross_entropy_loss(y_true: float, y_pred: float):
    return -y_true * np.log(y_pred)


def cross_entropy_loss_derivative(y_true: float, y_pred: float):
    return - y_true/y_pred


if __name__ == "__main__":
    """set_random_seed(123)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    layers = [FullyConnected(input_size=784, output_size=16), Tanh(), FullyConnected(input_size=16, output_size=16), Tanh(), FullyConnected(input_size=16, output_size=10), Tanh()]
    loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape(((10000, 28 * 28)))
    x_test = x_test.astype('float32') / 255

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)
    print(net.get_parameters().shape)"""


    """print("Loss", net.combined_loss(x_train, y_train))

    results = []
    for attributes in x_test:
        results.append(np.argmax(net(attributes)))
    accuracy = sum([1 for prediction, label in zip(results, y_test) if prediction == label]) / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))"""

    iris = load_iris()

    x = iris.data
    x = x.astype("float32") / np.max(x)
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)


    # Small test model
    layers = [FullyConnected(input_size=4, output_size=4), Tanh(),
              FullyConnected(input_size=4, output_size=4), Tanh(),
              FullyConnected(input_size=4, output_size=3), Tanh()]
    loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    net.fit(x_train, y_train, 5, 0.1)
    #print(net.get_parameters().shape)

    results = []
    for attributes in x_test:
        results.append(np.argmax(net(attributes)))
    accuracy = sum([1 for prediction, label in zip(results, y_test) if
                    prediction == label]) / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))
