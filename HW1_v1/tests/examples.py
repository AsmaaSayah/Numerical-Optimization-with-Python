import numpy as np

def quadratic_example(x, hessian=False):
    if hessian:
        return np.array([[2, 0], [0, 2]])
    return np.sum(x ** 2)

def rosenbrock(x, hessian=False):
    if hessian:
        return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def linear_example(x, hessian=False):
    if hessian:
        return np.zeros((2, 2))
    a = np.array([1, 2])  # Choose any nonzero vector
    return np.dot(a, x)

def custom_function(x1, x2, hessian=False):
    if hessian:
        return np.zeros((2, 2))
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)

print('ok')