from typing import List
import numpy as np
from layers import Layer
from sklearn.utils import shuffle

def min_max_norm(val, min_val, max_val, new_min, new_max):
  return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)

        # Normalize
        #x = np.array([min_max_norm(pred, -1, 1, 1e-10, 1) for pred in x])

        return x

    def compile(self, loss: Loss) -> None:
        self.loss_function = loss

    def get_parameters(self) -> np.ndarray:
        parameters_lists = []
        for layer in self.layers:
            if layer.get_weights_and_biases() is not None:
                parameters_lists.append(layer.get_weights_and_biases().flatten())

        return np.concatenate(parameters_lists)

    def split_parameters(self, combined_parameters: np.ndarray) -> list:
        combined_indexes = 0
        split_indices = []
        for layer in self.layers:
            if layer.get_weights_and_biases() is not None:
                split_indices.append(combined_indexes+(layer.input_size+1)*layer.output_size)
                combined_indexes += (layer.input_size+1)*layer.output_size

        parameters = np.split(combined_parameters, split_indices)
        reshaped_parameters = []
        layers = [layer for layer in self.layers if
                  layer.get_weights_and_biases() is not None]

        for layer, parameter_matrix in zip(layers, parameters):
            reshaped_parameters.append(parameter_matrix.reshape((layer.output_size, (layer.input_size+1))))
        return reshaped_parameters

    def get_gradients(self) -> np.ndarray:
        parameters_lists = []
        for layer in self.layers:
            if layer.get_gradient() is not None:
                gradients = layer.get_gradient()
                parameters_lists.append(gradients.flatten())

        return np.concatenate(parameters_lists)

    def split_gradients(self, combined_parameters: np.ndarray) -> list:
        combined_indexes = 0
        split_indices = []
        for layer in self.layers:
            if layer.get_gradient() is not None:
                split_indices.append(combined_indexes + (
                            layer.input_size + 1) * layer.output_size)
                combined_indexes += (layer.input_size + 1) * layer.output_size

        return np.split(combined_parameters, split_indices)

    def combined_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        results = []
        for single_x, single_y in zip(x, y):
            pred = self(single_x)
            results.append(self.loss_function.loss(single_y, pred))
        return sum(results) / len(results)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            verbose: int = 0) -> None:
        """Fit the network to the training data"""

        batches = len(x_train) // batch_size

        for epoch in range(epochs):
            # Shuffle the data
            x_train, y_train = shuffle(x_train, y_train)

            for index in range(batches):
                start_index = index * batch_size
                end_index = (index + 1) * batch_size

                x_batch = x_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]

                for layer in self.layers:
                    if layer.get_gradient() is not None:
                        layer.reset_gradients()

                gradient = self.get_gradients()
                step_loss_value = 0

                for x, y in zip(x_batch, y_batch):
                    # Reset all gradients
                    for layer in self.layers:
                        if layer.get_gradient() is not None:
                            layer.reset_gradients()

                    # Forward pass
                    x = self(x)

                    # Backward pass
                    """true_probability_dist = np.zeros((len(np.unique(y_train)),))
                    true_probability_dist[y] = 1"""
                    y = np.array([y])
                    loss_array = self.loss_function.loss(y, x)
                    step_loss_value += sum(loss_array)

                    loss_derivative = self.loss_function.loss_derivative(
                        y, x)
                    for layer in reversed(self.layers):
                        loss_derivative = layer.backward(loss_derivative)

                    gradient += self.get_gradients()

                loss_value = step_loss_value / len(x_batch)
                print(f"Epoch: {epoch}, Loss: {loss_value}")

                # Optimization
                weights = self.get_parameters()
                gradient = gradient / len(x_batch)
                weights -= learning_rate * gradient

                # Assigning new weights and biases
                splited_weights = self.split_parameters(weights)
                layers = [layer for layer in self.layers if
                          layer.get_gradient() is not None]
                for layer, new_weigts in zip(layers, splited_weights):
                    layer.parameters = new_weigts




