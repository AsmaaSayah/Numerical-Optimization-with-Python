# src/utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_contour(obj_func, xlims, ylims, path=None, labels=None):
    """
    Plot contour lines of the objective function.

    Parameters:
    - obj_func: The objective function.
    - xlims: Limits for the x-axis.
    - ylims: Limits for the y-axis.
    - path: Path of iterations (optional).
    - labels: Labels for the paths (optional).
    """
    x = np.linspace(xlims[0], xlims[1], 100)
    y = np.linspace(ylims[0], ylims[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = obj_func([X, Y])
    plt.contour(X, Y, Z, levels=20)
    if path:
        for p, label in zip(path, labels):
            plt.plot(p[:, 0], p[:, 1], label=label)
        plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Plot of Objective Function')
    plt.show()

def plot_function_values(iter_values):
    """
    Plot function values at each iteration for given methods.

    Parameters:
    - iter_values: Dictionary containing function values for each method.
    """
    for label, values in iter_values.items():
        plt.plot(range(len(values)), values, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Values vs. Iteration')
    plt.legend()
    plt.show()
