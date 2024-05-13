import numpy as np
import matplotlib.pyplot as plt

def plot_contour(obj_func, xlims, ylims, paths=None, labels=None):
    """
    Plot contour lines of the objective function.

    Parameters:
    - obj_func: The objective function.
    - xlims: Limits for the x-axis.
    - ylims: Limits for the y-axis.
    - paths: List of paths (optional).
    - labels: List of labels for paths (optional).
    """
def plot_contour(obj_func, xlims, ylims, paths=None, labels=None):
    x = np.linspace(xlims[0], xlims[1], 100)
    y = np.linspace(ylims[0], ylims[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj_func([X[i, j], Y[i, j]])

    plt.contour(X, Y, Z, levels=20)

    if paths:
        for path, label in zip(paths, labels):
            plt.plot(path[:, 0], path[:, 1], label=label)

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
