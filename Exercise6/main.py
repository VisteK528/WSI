from layers import FullyConnected, Tanh, Softmax
from network import Loss, Network
import numpy as np
from keras.datasets import mnist
from keras.utils import set_random_seed
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def cross_entropy_loss(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    log_predicted_dist = np.array([-np.log(x) for x in predicted_dist])
    return np.multiply(expected_dist, log_predicted_dist)


def cross_entropy_loss_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return predicted_dist - expected_dist

def squared_error(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected-predicted, 2) for expected, predicted in zip(expected_dist, predicted_dist)])

def squared_error_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return 2*(predicted_dist - expected_dist)


if __name__ == "__main__":
    set_random_seed(123)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    one_hot = np.zeros((len(y_train), len(np.unique(y_test))))
    for i, y in enumerate(y_train):
        one_hot[i, y] = 1

    layers = [FullyConnected(input_size=784, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=10), Tanh()]

    loss = Loss(squared_error, squared_error_derivative)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape(((10000, 28 * 28)))
    x_test = x_test.astype('float32') / 255

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    net.fit(x_train, one_hot, 5, batch_size=64, learning_rate=0.1)

    results = []
    for attributes in x_test:
        results.append(np.argmax(net(attributes)))
    accuracy = sum([1 for prediction, label in zip(results, y_test) if prediction == label]) / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))

    """iris = load_iris()

    x = iris.data
    x = x.astype("float32") / np.max(x)
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


    # Small test model
    layers = [FullyConnected(input_size=4, output_size=40), Tanh(),
              FullyConnected(input_size=40, output_size=40), Tanh(),
              FullyConnected(input_size=40, output_size=40), Tanh(),
              FullyConnected(input_size=40, output_size=3), Tanh()]
    #loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)
    loss = Loss(squared_error, squared_error_derivative)



    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    net.fit(x_train, y_train, 1000, batch_size=18, learning_rate=0.1)

    predicted_sum = 0
    for attributes, label in zip(x_test, y_test):
        prediction_dist = net(attributes)
        print("Prediction_dist:", prediction_dist)
        predicted_label = np.argmax(prediction_dist)
        print("Predicted label:", predicted_label, ", Actual label:", label)
        if predicted_label == label:
            predicted_sum += 1

    accuracy = predicted_sum / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))"""
