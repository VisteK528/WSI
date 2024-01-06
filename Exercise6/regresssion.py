from layers import FullyConnected, Tanh, Linear, ReLU, Sigmoid
from network import Loss, Network
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from optimizers import MomentumSGD


def squared_error(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected-predicted, 2) for expected, predicted in zip(expected_dist, predicted_dist)])

def squared_error_derivative(expected_dist: np.ndarray, predicted_dist: np.ndarray) -> np.ndarray:
    return 2*(predicted_dist - expected_dist)


def f(x):
    return ((x**2) * np.sin(x)) + (100 * np.sin(x) * np.cos(x))


if __name__ == "__main__":
    test_range = np.arange(-5, 5, 0.1)
    x = test_range
    y = np.array([np.array([f(point)]) for point in x])
    x = np.array([np.array([float(element)]) for element in x])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    layers = [
        FullyConnected(input_size=1, output_size=16), Tanh(),
        FullyConnected(input_size=16, output_size=16), ReLU(),
        FullyConnected(input_size=16, output_size=8), Tanh(),
        FullyConnected(input_size=8, output_size=1), Linear()
    ]

    loss = Loss(squared_error, squared_error_derivative)
    optimizer = MomentumSGD(alpha=1e-2, beta=0.9)

    net = Network(layers)
    net.compile(loss, optimizer)

    net.fit(x_train, y_train, epochs=300, batch_size=len(x_train), verbose=1)

    y_pred = np.array([float(net(x_sample).reshape(1, )) for x_sample in
                       np.array([np.array([float(element)]) for element in
                                 test_range])])
    real = np.array([f(point) for point in test_range])

    plt.plot(test_range, real, label="Actual data")
    plt.plot(test_range, y_pred, label="Predicted data")
    plt.grid()
    plt.legend()
    plt.show()
