import numpy as np
import matplotlib.pyplot as plt


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


print("ok")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the quadratic objective function
def func_quad(x):
    return x[0]**2 + x[1]**2 + (x[2] + 1)**2

# Create a grid of points in 3D space
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
z = np.linspace(0, 1, 50)

# Initialize an empty grid for the objective function values
F = np.zeros((50, 50))

# Evaluate the objective function at each point in the grid
for i in range(50):
    for j in range(50):
        for k in range(50):
            point = [x[i], y[j], z[k]]
            F[i, j] = func_quad(point)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create coordinate arrays for the grid
X, Y = np.meshgrid(x, y)

# Plot the objective function surface
ax.plot_surface(X, Y, F, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Objective Function')
ax.set_title('Objective Function Surface')

# Show the plot
plt.show()

print("ok2")

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Define the objective function
# def objective_func(x, y):
#     return x + y
#
# # Define the inequality constraint functions
# def constraint_1(x, y):
#     return -x + y - 1
#
# def constraint_2(x, y):
#     return y - 1
#
# def constraint_3(x, y):
#     return x - 2
#
# def constraint_4(x, y):
#     return -y
#
# # Create a grid of points in the feasible region
# x = np.linspace(0, 2, 50)
# y = np.linspace(0, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the objective function and constraint functions at each point in the grid
# Z_obj = objective_func(X, Y)
# Z_constr_1 = constraint_1(X, Y)
# Z_constr_2 = constraint_2(X, Y)
# Z_constr_3 = constraint_3(X, Y)
# Z_constr_4 = constraint_4(X, Y)
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the objective function surface
# ax.plot_surface(X, Y, Z_obj, cmap='viridis', alpha=0.5)
#
# # Plot the inequality constraints
# ax.plot_surface(X, Y, Z_constr_1, color='r', alpha=0.3)
# ax.plot_surface(X, Y, Z_constr_2, color='g', alpha=0.3)
# ax.plot_surface(X, Y, Z_constr_3, color='b', alpha=0.3)
# ax.plot_surface(X, Y, Z_constr_4, color='m', alpha=0.3)
#
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Objective Function')
# ax.set_title('Objective Function and Constraints')
#
# # Show the plot
# plt.show()
#

print("ok3")
import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
# def objective_func(x, y):
#     return -x - y  # turned it to a min problem so i flipped the signs

# Define the inequality constraint functions
def constraint_1(x, y):
    return y - (-x + 1)

def constraint_2(x, y):
    return y - 1

def constraint_3(x, y):
    return x - 2

def constraint_4(x, y):
    return y

# Create a grid of points in the feasible region
x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Evaluate the objective function and constraint functions at each point in the grid
#Z_obj = objective_func(X, Y)

Z_constr_1 = constraint_1(X, Y)
Z_constr_2 = constraint_2(X, Y)
Z_constr_3 = constraint_3(X, Y)
Z_constr_4 = constraint_4(X, Y)

# Create a scatter plot for the constraints
fig, ax = plt.subplots()

# Plot the inequality constraints as scattered lines
ax.scatter(X[Z_constr_1 >= 0], Y[Z_constr_1 >= 0], color='r', label='Constraint 1')
ax.scatter(X[Z_constr_2 >= 0], Y[Z_constr_2 >= 0], color='g', label='Constraint 2')
ax.scatter(X[Z_constr_3 >= 0], Y[Z_constr_3 >= 0], color='b', label='Constraint 3')
ax.scatter(X[Z_constr_4 >= 0], Y[Z_constr_4 >= 0], color='m', label='Constraint 4')

# Plot the first constraint as a line
x_line = np.linspace(0, 3, 100)
y_line = 1 + x_line
ax.plot(x_line, y_line, 'k--', label='Constraint 1')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Constraints')
ax.legend()

# Create a contour plot for the feasible area
levels = [0]  # Contour level of 0 represents the feasible area
ax.contour(X, Y, Z_constr_1, levels, colors='gray', alpha=0.3)
ax.contour(X, Y, Z_constr_2, levels, colors='gray', alpha=0.3)
ax.contour(X, Y, Z_constr_3, levels, colors='gray', alpha=0.3)
ax.contour(X, Y, Z_constr_4, levels, colors='gray', alpha=0.3)

# Show the plot
plt.show()