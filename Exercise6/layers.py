from abc import abstractmethod, ABC
import numpy as np


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
        self._parameters_gradients = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: np.ndarray) -> None:
        self._parameters = parameters

    @property
    def parameters_gradients(self):
        return self._parameters_gradients

    @parameters_gradients.setter
    def parameters_gradients(self, parameters_gradients: np.ndarray) -> None:
        self._parameters_gradients = parameters_gradients

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
        self._parameters_gradients = np.zeros((output_size, input_size+1))
        self._last_input_feed_forward_values = None
        self._last_output_feed_forward_values = None

    def reset_gradients(self) -> None:
        self._parameters_gradients = np.zeros((self.output_size, self.input_size + 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.array([*x, 1])
        assert self._parameters.shape[1] == x.shape[0]
        y = self._parameters @ x

        self._last_input_feed_forward_values = x
        self._last_output_feed_forward_values = y
        return y

    def backward(self, output_derivative) -> np.ndarray:
        layer_gradient = output_derivative @ self._parameters[:, :-1]

        self._parameters_gradients = np.tile(self._last_input_feed_forward_values, (self._parameters.shape[0], 1))
        for i in range(self._parameters.shape[0]):
            self._parameters_gradients[i] *= output_derivative[i]

        return layer_gradient


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_input_feed_forward_values = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input_feed_forward_values = x
        return np.tanh(x)

    def backward(self, output_derivative) -> np.ndarray:
        return output_derivative @ np.ones(self._last_input_feed_forward_values.shape) - pow(np.tanh(self._last_input_feed_forward_values), 2)











