#utils.py
import numpy as np
import matplotlib.pyplot as plt


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



print("ok")