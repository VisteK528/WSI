from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np

class SimpleGradientDescent:
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-3, 2, 0.05)
    X, Y = np.meshgrid(X, Y)

    def __init__(self,
                 func: Callable[[float, float], float],
                 grad_func: Callable[[float, float], Tuple[float, float]],
                 alpha:float=0.1):
        self.alpha = alpha
        self.func = func
        self.grad_func = grad_func
        self.trace = None  # trace of search

    def _calc_Z_value(self):
        self.Z = self.func(self.X, self.Y)

    def plot_func(self):
        self._calc_Z_value()
        plt.figure()
        plt.contour(self.X, self.Y, self.Z, 50)
        if self.trace is not None:
            if len(self.trace)>0:
                plt.scatter(self.trace[:,0], self.trace[:,1], s=10)

    def calculate_func_vale(self, x1:float, x2:float) -> float:
        pass

    def calculate_func_grad(self, x1:float, x2:float) -> Tuple[float, float]:
        return self.grad_func(x1, x2)

    def gradient_descent_step(self, x1:float, x2:float) -> Tuple[float, float]:
        if self.trace is not None:
            return np.array(x1, x2) - self.alpha * np.array(self.calculate_func_grad(x1, x2))
        else:
            return (x1, x2)

    def minimize(self, x1_init:float, x2_init:float, steps:int, verbose:int=0, plot:bool=False)->float:
        for i in range(steps):
            if self.trace is None:
                result = self.gradient_descent_step(x1_init, x2_init)
                self.trace = np.empty((steps, 2))
                self.trace[i, :] = result
            else:
                result = self.gradient_descent_step(self.trace[i-1, 0], self.trace[i-1, 1])
                self.trace[i, :] = result

        self.trace = self.trace.reshape((2, steps))


class Function1:
    def __call__(self, X1: np.array, X2: np.array) -> np.array:
        return X1 ** 2 + X2 ** 2


class Grad1:
    def __call__(self, X1: np.array, X2: np.array) -> Tuple[np.array, np.array]:
        grad = (2 * X1, 2 * X2)
        return grad


if __name__ == "__main__":
    function1 = Function1()
    grad1 = Grad1()
    gradient = SimpleGradientDescent(function1, grad1)
    gradient.minimize(-2, -1.5, 10, 0)
    gradient.plot_func()