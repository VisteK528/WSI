from layers import FullyConnected, Tanh, Softmax, ReLU
from network import Loss, Network
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import (one_hot_encode, cross_entropy_loss, cross_entropy_loss_derivative,
                   squared_error, squared_error_derivative)
from optimizers import SimpleSGD, MomentumSGD


if __name__ == "__main__":
    iris = load_iris()

    x = iris.data
    x = x.astype("float32") / np.max(x)
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    y_train = one_hot_encode(y_train)

    # Small test model
    layers = [FullyConnected(input_size=4, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=16), Tanh(),
              FullyConnected(input_size=16, output_size=3), Tanh(),
              Softmax()]

    loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)
    #loss = Loss(squared_error, squared_error_derivative)
    optimizer_with_momentum = MomentumSGD(alpha=1e-2, beta=0.9)

    net = Network(layers)
    net.compile(loss, optimizer_with_momentum)

    #net.load_parameters("data/iris.npy")
    net.fit(x_train, y_train, 100, verbose=1, batch_size=18)
    #net.save_parameters("data/iris.npy")

    accuracy = net.evaluate(x_test, one_hot_encode(y_test))
    print(f"Accuracy: {accuracy*100:.2f}%")
