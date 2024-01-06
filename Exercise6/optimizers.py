import numpy as np
from layers import Layer
from abc import abstractmethod, ABC
from typing import List


class Optimizer(ABC):
    def __init__(self):
        # Network_architecture: Number of trainable layer in the network
        self._net_arch = None

    @property
    def net_arch(self):
        return self._net_arch

    @net_arch.setter
    def net_arch(self, net_architecture: int):
        self._net_arch = net_architecture

    @abstractmethod
    def optimize(self, layer: Layer, epoch: int):
        pass

    @abstractmethod
    def compile(self):
        pass


class SimpleSGD(Optimizer):
    def __init__(self, learning_rate: float, decay: float):
        super().__init__()
        assert 0 < learning_rate < 1
        self._start_lr = learning_rate
        self._lr = learning_rate
        self._decay = decay

    @property
    def learning_rate(self) -> float:
        return self._start_lr

    def compile(self):
        pass

    def optimize(self, layers: List[Layer], epoch: int):
        for layer in layers:
            if not layer.activation_layer:
                weights = layer.get_weights()
                biases = layer.get_biases()

                weights_gradient = layer.get_weights_gradient()
                biases_gradient = layer.get_biases_gradient()
                grad_nb = layer.accumulated_gradients

                new_weights = weights - self._lr * (weights_gradient / grad_nb)
                new_biases = biases - self._lr * (biases_gradient / grad_nb)

                layer.set_biases(new_biases)
                layer.set_weights(new_weights)


class MomentumSGD(Optimizer):
    def __init__(self, alpha: float, beta: float, decay: float):
        super().__init__()
        assert 0 < alpha < 1
        assert 0 < beta < 1
        self._start_alpha = alpha
        self._alpha = alpha
        self._beta = beta
        self._v = None

        self._decay = decay

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def compile(self):
        if self._net_arch is not None:
            self._v = [[None, None] for _ in range(self._net_arch)]

    def _calculate_velocity(self, v: list, weights_grad: np.ndarray, biases_grad: np.ndarray):
        new_velocities = []
        for vel, grad in zip(v, [weights_grad, biases_grad]):
            if vel is None:
                vel = self._alpha * grad
            else:
                vel = self._beta * vel + self._alpha * grad

            new_velocities.append(vel)

        return new_velocities

    def optimize(self, layers: List[Layer], epoch: int):
        assert type(self._net_arch) == int
        assert len(self._v) == self._net_arch

        # Decrease alpha / lr parameter
        self._alpha = self._start_alpha * np.exp(-self._decay*epoch)

        layers = [layer for layer in layers if not layer.activation_layer]

        new_velocities = []
        for i, x in enumerate(zip(layers, self._v)):
            layer, v = x


            weights = layer.get_weights()
            biases = layer.get_biases()

            grad_nb = layer.accumulated_gradients
            weights_gradient = layer.get_weights_gradient() / grad_nb
            biases_gradient = layer.get_biases_gradient() / grad_nb

            v = self._calculate_velocity(v, weights_gradient, biases_gradient)
            new_velocities.append(v)

            new_weights = weights - v[0]
            new_biases = biases - v[1]

            layer.set_biases(new_biases)
            layer.set_weights(new_weights)

        self._v = new_velocities



