from layers import FullyConnected, Tanh, Softmax
from network import Loss, Network
import numpy as np
from keras.datasets import mnist
from keras.utils import set_random_seed
from utils import one_hot_encode

def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def cross_entropy_loss(expected_dist: np.ndarray,
                       predicted_dist: np.ndarray) -> np.ndarray:
    array = np.array([-float(y) * np.log(float(x)) for x, y in
                      zip(predicted_dist, expected_dist)])
    return array


def cross_entropy_loss_derivative(expected_dist: np.ndarray,
                                  predicted_dist: np.ndarray) -> np.ndarray:
    # epsilon = np.finfo(np.float64).eps
    array = np.array([x - y for x, y in zip(predicted_dist, expected_dist)])
    return array


def squared_error(expected_dist: np.ndarray,
                  predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected - predicted, 2) for expected, predicted in
                     zip(expected_dist, predicted_dist)])


def squared_error_derivative(expected_dist: np.ndarray,
                             predicted_dist: np.ndarray) -> np.ndarray:
    return 2 * (predicted_dist - expected_dist)


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

    loss = Loss(squared_error, squared_error_derivative)
    loss2 = Loss(cross_entropy_loss, cross_entropy_loss_derivative)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape(((10000, 28 * 28)))
    x_test = x_test.astype('float32') / 255

    net = Network(layers2, learning_rate=0.1)
    net.compile(loss2)

    net.fit(x_train, y_train, 5, verbose=1, batch_size=128, learning_rate=0.3)

    results = []
    for attributes in x_test:
        results.append(np.argmax(net(attributes)))
    accuracy = sum([1 for prediction, label in zip(results, y_test) if
                    prediction == label]) / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))
