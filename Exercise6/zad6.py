from abc import abstractmethod, ABC
from typing import List
import numpy as np
from keras.datasets import mnist
from keras.utils import set_random_seed


def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01
        self._parameters = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: np.ndarray) -> None:
        self._parameters = parameters

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) ->np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size:int, output_size:int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self._parameters = np.random.uniform(-1, 1, (output_size, input_size+1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.array([*x, 1])
        assert self._parameters.shape[1] == x.shape[0]
        y = self._parameters @ x
        return y

    def backward(self, output_error_derivative) -> np.ndarray:
        pass


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, output_error_derivative)->np.ndarray:
        return 1 - pow(np.tanh(output_error_derivative), 2)


class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function for a particular x"""
        return self.loss_function(x, y)

    def loss_derivative(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        """Loss function derivative for a particular x and y"""
        return self.loss_function(x, y)


class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_function = None
        self._network_parameters = None

    def compile(self, loss: Loss) -> None:
        self.loss_function = loss
        self._network_parameters = np.array([layer.parameters for layer in self.layers if layer.parameters is not None])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def combined_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        results = []
        for single_x, single_y in zip(x, y):
            pred = self(single_x)
            results.append(self.loss_function.loss(single_y, pred))
        return sum(results) / len(results)

    def backpropagate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Backpropagation to update model parameters"""
        # Forward pass
        layer_outputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            layer_outputs.append(x)

        # Backward pass
        loss_derivative = self.loss_function.loss_derivative(x, y)
        for layer, layer_output in zip(reversed(self.layers), reversed(layer_outputs[:-1])):
            loss_derivative = layer.backward(loss_derivative, layer_output)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            verbose: int = 0) -> None:
        """Fit the network to the training data"""

        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                self.backpropagate(x, y)


def cross_entropy_loss(y_true: float, y_pred: float):
    return -y_true * np.log(y_pred)


def cross_entropy_loss_derivative(y_true: float, y_pred: float):
    return - y_true/y_pred


if __name__ == "__main__":
    set_random_seed(123)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    layers = [FullyConnected(input_size=784, output_size=784), Tanh(), FullyConnected(input_size=784, output_size=16), Tanh(), FullyConnected(input_size=16, output_size=10), Tanh()]
    loss = Loss(cross_entropy_loss, cross_entropy_loss_derivative)

    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape(((10000, 28 * 28)))
    x_test = x_test.astype('float32') / 255

    net = Network(layers, learning_rate=0.1)
    net.compile(loss)
    """print("Loss", net.combined_loss(x_train, y_train))

    results = []
    for attributes in x_test:
        results.append(np.argmax(net(attributes)))
    accuracy = sum([1 for prediction, label in zip(results, y_test) if prediction == label]) / len(y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))"""
