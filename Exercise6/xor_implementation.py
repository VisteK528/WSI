from layers import FullyConnected, Tanh, Softmax, ReLU, Linear
from network import Loss, Network
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import (one_hot_encode, cross_entropy_loss, cross_entropy_loss_derivative,
                   squared_error, squared_error_derivative)


if __name__ == "__main__":
    iris = load_iris()

    x = np.random.randint(low=0, high=2, size=(1000, 2))
    y = np.array([attributes[0] ^ attributes[1] for attributes in x])

    print(x[:10])
    print(y[:10])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    y_train = one_hot_encode(y_train)

    # Small test model
    layers = [FullyConnected(input_size=2, output_size=2), Tanh(),
              FullyConnected(input_size=2, output_size=1), Tanh(),
              Linear()]

    #loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)
    loss = Loss(squared_error, squared_error_derivative)

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)

    #net.load_parameters("data/iris.npy")
    #net.fit(x_train, y_train, 1, verbose=1, batch_size=10, learning_rate=0.3)
    #net.save_parameters("data/iris.npy")

    accuracy = net.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")
