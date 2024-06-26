import numpy as np
import matplotlib.pyplot as plt



def quadratic(x):
    return x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2


def example_quadratic():
    ineq_constraints = [lambda x: - x[0],  # x >= 0
                        lambda x: - x[1],  # y >= 0
                        lambda x: - x[2]]  # z >= 0
    eq_constraints_mat = np.array([1, 1, 1])  # x+y+z
    eq_constraints_rhs = np.array([1])  # = 1
    x0 = np.array([0.1, 0.2, 0.7])
    return quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0


def linear(x):
    return -1 * x[0] - 1 * x[1]


def example_linear():
    ineq_constraints = [lambda x: -x[1],  # y >= 0
                        lambda x: x[0] - 2,  # x <= 2
                        lambda x: x[1] - 1,  # y <= 1
                        lambda x: -x[1] -x[0] + 1]  # y + x >= 1
    eq_constraints_mat = None
    eq_constraints_rhs = None
    x0 = np.array([0.5, 0.75])
    return linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0



################ HW1

def example_func_quad_1(X, en_hessian):
    # circle example

    # circle example
    Q = np.array([[1, 0], [0, 1]])
    func_x = X @ Q @ X

    grad_x = 2 * Q @ X

    hessian_x = []
    if en_hessian:
        hessian_x = 2 * Q

    return func_x, grad_x, hessian_x


def example_func_quad_2(X, en_hessian):
    # ellipse example
    Q = np.diag([1, 100])
    func_x = X @ Q @ X

    grad_x = 2 * Q @ X

    hessian_x = None
    if en_hessian:
        hessian_x = 2 * Q

    return func_x, grad_x, hessian_x


def example_func_quad_3(X, en_hessian):
    # rotated ellipse example
    Core = np.diag([100, 1])
    RotMat = np.array([[0.5 * np.sqrt(3), -0.5],
                       [0.5, 0.5 * np.sqrt(3)]])
    Q = RotMat.T @ Core @ RotMat

    func_x = X @ Q @ X
    grad_x = 2 * Q @ X

    hessian_x = None
    if en_hessian:
        hessian_x = 2 * Q

    return func_x, grad_x, hessian_x


def example_func_rosenbrock(X, en_hessian):
    # Rosenbrock example
    func_x = (1 - X[0]) ** 2 + 100. * (X[1] - X[0] ** 2) ** 2

    grad_x = np.array([400 * X[0] ** 3 - 400 * X[0] * X[1] + 2 * X[0] - 2,
                       -200 * X[0] ** 2 + 200 * X[1]])

    hessian_x = None
    if en_hessian:
        hessian_x = np.array([[1200 * X[0] ** 2 - 400 * X[1] + 2, -400 * X[0]],
                              [-400 * X[0], 200]])

    return func_x, grad_x, hessian_x


def example_func_linear(X, en_hessian):
    # linear example
    a = np.ones(len(X))

    func_x = a @ X
    grad_x = a
    hessian_x = None if not en_hessian else np.zeros((len(X), len(X)))

    return func_x, grad_x, hessian_x


def example_func_nonquad(X, en_hessian):
    # non quadratic exponential example
    func_x = np.exp(X[0] + 3 * X[1] - 0.1) + np.exp(X[0] - 3 * X[1] - 0.1) + np.exp(-X[0] - 0.1)

    grad_x = np.array([np.exp(X[0] + 3 * X[1] - 0.1) + np.exp(X[0] - 3 * X[1] - 0.1) - np.exp(-X[0] - 0.1),
                       3 * np.exp(X[0] + 3 * X[1] - 0.1) - 3 * np.exp(X[0] - 3 * X[1] - 0.1)])

    hessian_x = None
    if en_hessian:
        hessian_x = np.array([[np.exp(X[0] + 3 * X[1] - 0.1) + np.exp(X[0] - 3 * X[1] - 0.1) + np.exp(-X[0] - 0.1),
                               3 * np.exp(X[0] + 3 * X[1] - 0.1) - 3 * np.exp(X[0] - 3 * X[1] - 0.1)],
                              [3 * np.exp(X[0] + 3 * X[1] - 0.1) - 3 * np.exp(X[0] - 3 * X[1] - 0.1),
                               9 * np.exp(X[0] + 3 * X[1] - 0.1) + 9 * np.exp(X[0] - 3 * X[1] - 0.1)]])

    return func_x, grad_x, hessian_x



