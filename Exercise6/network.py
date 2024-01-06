from typing import List
import numpy as np
from layers import Layer
from sklearn.utils import shuffle
from utils import printProgressBar
from optimizers import Optimizer
import time


class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function for a particular x"""
        return self.loss_function(x, y)

    def loss_sum(self, x: np.ndarray, y: np.ndarray) -> float:
        return sum(self.loss(x, y))

    def loss_derivative(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        """Loss function derivative for a particular x and y"""
        return self.loss_function_derivative(x, y)


class Network:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
        self.loss_function = None
        self._network_parameters = None
        self._optimizer = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        self.loss_function = loss
        self._optimizer = optimizer

        self._optimizer.net_arch = len([layer for layer in self.layers if not layer.activation_layer])
        self._optimizer.compile()

    def get_parameters(self) -> np.ndarray:
        parameters_lists = []
        for layer in self.layers:
            if not layer.activation_layer:
                parameters_lists.append(layer.get_weights_and_biases().flatten())

        return np.concatenate(parameters_lists)

    def split_parameters(self, combined_parameters: np.ndarray) -> list:
        combined_indexes = 0
        split_indices = []
        for layer in self.layers:
            if not layer.activation_layer:
                split_indices.append(combined_indexes+(layer.input_size+1)*layer.output_size)
                combined_indexes += (layer.input_size+1)*layer.output_size

        parameters = np.split(combined_parameters, split_indices)
        reshaped_parameters = []
        layers = [layer for layer in self.layers if not layer.activation_layer]

        for layer, parameter_matrix in zip(layers, parameters):
            reshaped_parameters.append(parameter_matrix.reshape((layer.output_size, (layer.input_size+1))))
        return reshaped_parameters

    def save_parameters(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            for layer in self.layers:
                np.save(f, layer.get_weights())
                np.save(f, layer.get_biases())
            f.close()

    def load_parameters(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            for layer in self.layers:
                layer.set_weights(np.load(f, allow_pickle=True))
                layer.set_biases(np.load(f, allow_pickle=True))
            f.close()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        samples = len(y_test)
        results = [np.argmax(self(attributes)) for attributes in x_test]

        accuracy = sum([1 for prediction, expected_dist in zip(results, y_test) if
                        prediction == np.argmax(expected_dist)]) / samples
        return accuracy

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int,
            verbose: int = 0) -> dict:

        """Fit the network to the training data"""

        batches = len(x_train) // batch_size
        all_loss_history = {}
        all_steps = 0

        for epoch in range(epochs):
            # Shuffle the data
            x_train, y_train = shuffle(x_train, y_train)

            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs}")

            for index in range(batches):
                start_index = index * batch_size
                end_index = (index + 1) * batch_size

                x_batch = x_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]

                for layer in self.layers:
                    if not layer.activation_layer:
                        layer.reset_gradients()

                step_loss_value = 0

                for x, y in zip(x_batch, y_batch):
                    # ============== Forward pass =============================
                    x = self(x)

                    # ============== Backward pass ============================

                    # Calculate the loss value for the step
                    value = self.loss_function.loss_sum(y, x)
                    step_loss_value += value

                    # Calculate the derivative of the loss function
                    # with respond to last forward pass outputs
                    loss_derivative = self.loss_function.loss_derivative(y, x)

                    # Backpropagation
                    for layer in reversed(self.layers):
                        loss_derivative = layer.backward(loss_derivative)

                loss_value = step_loss_value / len(x_batch)
                batch_accuracy = self.evaluate(x_batch, y_batch)

                all_loss_history[all_steps] = loss_value
                all_steps += 1

                self._optimizer.optimize(self.layers)

                if verbose == 1:
                    printProgressBar(index + 1, batches,
                                     prefix=f"{index+1}/{batches}",
                                     suffix=f"Loss: {loss_value:.4f}\tAccuracy: {batch_accuracy*100:.2f}%",
                                     fill="=")
                    time.sleep(0.02)

        return all_loss_history