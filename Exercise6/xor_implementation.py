from layers import FullyConnected, Tanh, Softmax, ReLU, Linear
from network import Loss, Network
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import (one_hot_encode, cross_entropy_loss, cross_entropy_loss_derivative,
                   squared_error, squared_error_derivative)
from optimizers import SimpleSGD


if __name__ == "__main__":
    iris = load_iris()

    x = np.random.randint(low=0, high=2, size=(1000, 2))
    y = np.array([attributes[0] ^ attributes[1] for attributes in x])

    print(x[:10])
    print(y[:10])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    #y_train = one_hot_encode(y_train)
    y_train = np.array([np.array([y]) for y in y_train])

    # Small test model
    layers = [FullyConnected(input_size=2, output_size=2), Tanh(),
              FullyConnected(input_size=2, output_size=1), Tanh()]

    loss = Loss(squared_error, squared_error_derivative)
    optimizer = SimpleSGD(learning_rate=0.1, decay=0.1)

    net = Network(layers)
    net.compile(loss, optimizer)
    net.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1)

    y_test = [np.array(y) for y in y_test]
    for x, y in zip(x_test[:10], y_test[:10]):
        pred = net(x)
        print(y, pred)
