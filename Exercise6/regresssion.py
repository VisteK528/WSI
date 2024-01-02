from layers import FullyConnected, Tanh, Linear
from network import Loss, Network
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def squared_error(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected-predicted,2) for expected, predicted in zip(expected_dist, predicted_dist)])

def squared_error_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return 2*(predicted_dist - expected_dist)


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.01)
    y = np.array([np.sin(point) for point in x])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

    layers = [FullyConnected(input_size=1, output_size=1), Tanh(),
              FullyConnected(input_size=1, output_size=1), Tanh(),
              FullyConnected(input_size=1, output_size=1), Tanh(),
              FullyConnected(input_size=1, output_size=1), Linear()]

    loss = Loss(squared_error, squared_error_derivative)

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    net.fit(x_train, y_train, 50, batch_size=1, learning_rate=0.1)

    y_pred = np.array([float(net(x_sample)) for x_sample in np.arange(-5, 5, 0.01)])
    real = np.array([np.sin(point) for point in np.arange(-5, 5, 0.01)])

    plt.plot(np.arange(-5, 5, 0.01), real, label="Actual data")
    plt.plot(np.arange(-5, 5, 0.01), y_pred, label="Predicted data")
    plt.grid()
    plt.legend()
    plt.show()
