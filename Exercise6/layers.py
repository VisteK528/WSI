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
        self._biases = None
        self._parameters_gradients = None

    def get_weights_and_biases(self) -> np.ndarray:
        return self._parameters

    def set_weights_and_biases(self, parameters: np.ndarray) -> None:
        pass

    def get_gradient(self) -> np.ndarray:
        pass

    def set_gradient(self, parameters_gradients: np.ndarray) -> None:
        return self._parameters_gradients

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

        self._parameters = np.random.uniform(-1/np.sqrt(self.input_size), 1/np.sqrt(self.input_size), (self.output_size, self.input_size))
        self._biases = np.random.uniform(-1/np.sqrt(self.input_size), 1//np.sqrt(self.input_size), (self.output_size,))

        self._parameters_gradients = np.zeros((self.output_size, self.input_size+1))
        self._last_a = None

    def reset_gradients(self) -> None:
        self._parameters_gradients = np.zeros((self.output_size, self.input_size + 1))

    def get_weights_and_biases(self) -> np.ndarray:
        return np.concatenate((self._parameters, self._biases.reshape(-1, 1)), axis=1)

    def set_weights_and_biases(self, parameters: np.ndarray) -> None:
        self._parameters = parameters[:, :-1]
        self._biases = parameters[:, -1]

    def get_gradient(self) -> np.ndarray:
        return self._parameters_gradients

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.dot(self._parameters, x)
        y += + self._biases

        self._last_a = x
        return y

    def backward(self, output_derivative) -> np.ndarray:
        # Calculate derivative with respect to previous layer activation
        layer_gradient = np.dot(np.transpose(self._parameters), output_derivative)

        # Calculate derivative with respect to current layer weights
        self._parameters_gradients[:, :-1] = np.dot(output_derivative[:, np.newaxis], np.transpose(self._last_a[:, np.newaxis]))

        # Calculate derivative with respect to current layer biases
        self._parameters_gradients[:, -1] = output_derivative.reshape(output_derivative.shape[0],)

        return layer_gradient


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_z = x
        return np.tanh(x)

    def backward(self, output_derivative) -> np.ndarray:
        gradients = np.multiply(output_derivative, np.array([1 - pow(np.tanh(x), 2) for x in self._last_z]))
        return gradients


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_z = x
        x = np.exp(x - np.max(x))
        denominator = sum([np.exp(y) for y in x])
        return np.array([np.exp(element) / denominator for element in x])

    def backward(self, output_derivative) -> np.ndarray:
        gradients = np.multiply(output_derivative, np.array([x-pow(x,2) for x in self._last_z]))
        return gradients


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_z = x
        return np.array([max(0, element) for element in x])

    def backward(self, output_derivative) -> np.ndarray:
        gradients = np.multiply(output_derivative, np.where(self._last_z < 0, 0, 1))
        return gradients


class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_z = x
        val_exp = 1 / (1 + np.exp(-x))
        return np.array(val_exp)

    def backward(self, output_derivative) -> np.ndarray:
        val_exp = 1 / (1 + np.exp(-self._last_z))
        local_gradient = np.array(val_exp * (1 - val_exp))
        gradients = np.multiply(output_derivative, local_gradient)
        return gradients


class Linear(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._last_z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_z = x
        return x

    def backward(self, output_derivative) -> np.ndarray:
        gradients = np.multiply(output_derivative, np.ones(self._last_z.shape))
        return gradients











