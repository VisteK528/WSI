from layers import FullyConnected, Tanh, Softmax
from network import Loss, Network
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import one_hot_encode


def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def cross_entropy_loss(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    array = np.array([-float(y)*np.log(float(x)) for x, y in zip(predicted_dist, expected_dist)])
    return array


def cross_entropy_loss_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    # epsilon = np.finfo(np.float64).eps
    array = np.array([x - y for x, y in zip(predicted_dist, expected_dist)])
    return array

def squared_error(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected-predicted, 2) for expected, predicted in zip(expected_dist, predicted_dist)])

def squared_error_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return 2*(predicted_dist - expected_dist)


if __name__ == "__main__":
    iris = load_iris()

    x = iris.data
    x = x.astype("float32") / np.max(x)
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

    y_train = one_hot_encode(y_train)

    # Small test model
    layers = [FullyConnected(input_size=4, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=3), Tanh(),
              Softmax()]
    loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)
    #loss = Loss(squared_error, squared_error_derivative)



    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    net.fit(x_train, y_train, 10, verbose=1, batch_size=10, learning_rate=0.1)

    predicted_sum = 0
    for attributes, label in zip(x_test, y_test):
        prediction_dist = net(attributes)
        print("Prediction_dist:", prediction_dist)
        predicted_label = np.argmax(prediction_dist)
        print("Predicted label:", predicted_label, ", Actual label:", label)
        if predicted_label == label:
            predicted_sum += 1

    accuracy = predicted_sum / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))