#utils.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


############ HW2

# Plot the path taken by the algorith
def plot_path(path):
    path = np.array(path)
    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'o-')
    plt.title('Path taken by the algorithm')
    plt.show()


# Plot the objective value vs. outer iteration number
def plot_obj_vs_iter(objectives):
    plt.figure()
    plt.plot(objectives)
    plt.title('Objective value vs. iteration number')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective value')
    plt.show()


def plot_path_3d(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'o-', label='Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_path_2d(path):
    fig, ax = plt.subplots()
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'o-', label='Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

def plot_feasible_region_3d(path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z = np.maximum(Z, 0)
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.5, color='gray', label='Feasible Region')
    
    # Plot the path if provided
    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'o-', label='Path')
    
    # Add legends and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Customize legend if needed
    # ax.legend(['Feasible Region', 'Path'])

    plt.show()

def plot_feasible_region_2d():
    fig, ax = plt.subplots()
    x = np.linspace(-1, 3, 400)
    colors = ['blue', 'green', 'red', 'purple']
    # y >= -x + 1
    ax.plot(x, -x + 1, color=colors[0], label='y >= -x + 1')
    # y <= 1
    ax.plot(x, np.ones_like(x), color=colors[1], label='y <= 1')
    # x <= 2
    ax.axvline(x=2, color=colors[2], label='x <= 2')
    # y >= 0
    ax.axhline(y=0, color=colors[3], label='y >= 0')
    
    # Fill feasible region
    y1 = -x + 1
    y2 = np.ones_like(x)
    y3 = np.zeros_like(x)
    y4 = np.minimum(y1, y2)
    ax.fill_between(x, y3, y4, where=(y3 < y4), color='gray', alpha=0.5, label='Feasible Region')

    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

def plot_objective_vs_iteration(objectives):
    plt.figure()
    plt.plot(objectives, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.show()


########### HW1

def draw_results(ax, results, color, marker, label):
    if results:
        ax.scatter(results[0][0], results[0][1], results[1], c=color, s=100, marker=marker, label=label)
        ax.plot(results[2][:, 0], results[2][:, 1], results[3], c=color)


def plot_contour(obj_func, func_name, results_gd, results_newton):
    # Color and marker modifications
    colors = ['crimson', 'seagreen', 'royalblue', 'purple']
    markers = ['o', 'v', '^', '<']

    # Defining the function values at each point in the meshgrid
    x_values = np.linspace(-10., 10., 100)
    y_values = np.linspace(-10., 10., 100)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)

    func_mesh_values = np.zeros(mesh_x.shape)
    for i in range(mesh_x.shape[0]):
        for j in range(mesh_x.shape[1]):
            func_mesh_values[i, j], _, _ = obj_func(np.array([mesh_x[i, j], mesh_y[i, j]]), en_hessian=False)

    # Creating the figure
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.contour3D(mesh_x, mesh_y, func_mesh_values, 60, cmap='magma')

    # Plotting the results
    for result, color, marker in zip([results_gd, results_newton], colors, markers):
        draw_results(ax, result, color, marker, result[5] if result else "")

    # Formatting the plot
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('$f(x)$')
    ax.set_title(f'{func_name} Minimization Visualization')
    ax.view_init(elev=45, azim=60)
    plt.legend()
    plt.show()


def plot_function_values(func_name, results_gd, results_newton):
    if func_name != 'linear': 
        results_gd_tmp = results_gd[2][:, 1]
        results_newton_tmp = results_newton[2][:, 1]
        plt.figure(figsize=(10, 6))
        
        # Plotting function values vs. iteration number for Gradient Descent
        plt.plot(range(len(results_gd_tmp)), results_gd_tmp, label='Gradient Descent')
        
        # Plotting function values vs. iteration number for Newton's method
        plt.plot(range(len(results_newton_tmp)), results_newton_tmp, label="Newton's Method")
        
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title(f'Function Values vs. Iteration Number for {func_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        results_gd_tmp = results_gd[2][:, 1]
        plt.figure(figsize=(10, 6))
        # Plotting function values vs. iteration number for Gradient Descent
        plt.plot(range(len(results_gd_tmp)), results_gd_tmp, label='Gradient Descent')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title(f'Function Values vs. Iteration Number for {func_name}')
        plt.legend()
        plt.grid(True)
        plt.show()  



