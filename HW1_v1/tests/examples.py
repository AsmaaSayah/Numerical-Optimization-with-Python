import numpy as np

def quadratic_example(x, compute_hessian=True, example_num=1):
    """Quadratic example objective function."""
    if example_num == 1:
        Q = np.array([[1, 0], [0, 1]])
    elif example_num == 2:
        Q = np.array([[1, 0], [0, 100]])
    elif example_num == 3:
        sqrt3 = np.sqrt(3)
        Q = np.array([[sqrt3/2, -0.5], [0.5, sqrt3/2]]) @ np.array([[100, 0], [0, 1]]) @ np.array([[sqrt3/2, -0.5], [0.5, sqrt3/2]])
    else:
        raise ValueError("Invalid example number")

    f = np.dot(x.T, Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if compute_hessian else None
    print(f)
    return f, g, h

def rosenbrock_example(x, compute_hessian=True):
    """Rosenbrock example objective function."""
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([
        -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])
    h = None  # Hessian not needed for Rosenbrock function
    return f, g, h

def linear_example(x, a=np.array([1, 1])):
    """Linear example objective function."""
    f = np.dot(a.T, x)
    g = a
    h = None  # Hessian not needed for linear function
    return f, g, h

def smoothed_corner_triangles(x):
    """Smoothed corner triangles example objective function."""
    f = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    ])
    h = None  # Hessian not needed for smoothed corner triangles function
    return f, g, h
