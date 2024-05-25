import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
import time


def quadratic_model(Y, f_values):
    """
    Construct a quadratic model based on interpolation points and function values.
    
    Parameters:
    Y (list of np.ndarray): Interpolation points.
    f_values (np.ndarray): Function values at the interpolation points.
    
    Returns:
    np.ndarray: Coefficients of the quadratic model.
    """
    q = len(Y)
    n = len(Y[0])
    required_points = (n + 1) * (n + 2) // 2
    if q < required_points:
        raise ValueError(f"Not enough interpolation points. Need at least {required_points}, but got {q}.")

    A = np.zeros((q, required_points))
    for i in range(q):
        row = [1] + list(Y[i]) + [Y[i][j] * Y[i][k] for j in range(n) for k in range(j, n)]
        A[i, :] = row
    
    coefs = solve(A, f_values)
    return coefs

def model_value(coefs, x):
    """
    Evaluate the quadratic model at a given point.
    
    Parameters:
    coefs (np.ndarray): Coefficients of the quadratic model.
    x (np.ndarray): Point at which to evaluate the model.
    
    Returns:
    float: Value of the quadratic model at x.
    """
    n = len(x)
    terms = [1] + list(x) + [x[j] * x[k] for j in range(n) for k in range(j, n)]
    return np.dot(coefs, terms)

def trust_region_subproblem(coefs, xk, delta):
    """
    Solve the trust region subproblem to find the step.
    
    Parameters:
    coefs (np.ndarray): Coefficients of the quadratic model.
    xk (np.ndarray): Current point.
    delta (float): Trust region radius.
    
    Returns:
    np.ndarray: Step to take within the trust region.
    """
    def model_p(p):
        return model_value(coefs, xk + p)
    
    bounds = [(-delta, delta)] * len(xk)
    res = minimize(model_p, np.zeros_like(xk), bounds=bounds)
    return res.x

def compute_rho(f, xk, pk, mk, f_values, coefs):
    """
    Compute the ratio of actual reduction to predicted reduction.
    
    Parameters:
    f (function): Objective function.
    xk (np.ndarray): Current point.
    pk (np.ndarray): Step taken.
    mk (float): Value of the model at the current point.
    f_values (np.ndarray): Function values at the interpolation points.
    coefs (np.ndarray): Coefficients of the quadratic model.
    
    Returns:
    float: Ratio ρ of actual to predicted reduction.
    """
    
    actual_reduction = f(xk) - f(xk + pk)
    predicted_reduction = mk - model_value(coefs, xk + pk)
    return actual_reduction / predicted_reduction if predicted_reduction != 0 else 0

def geometry_improving_procedure(Y):
    """
    Improve the geometry of the interpolation set.
    
    Parameters:
    Y (list of np.ndarray): Current interpolation set.
    
    Returns:
    list of np.ndarray: Updated interpolation set with improved geometry.
    """
    perturbed_Y = Y.copy()
    perturbed_Y[0] = Y[0] + 0.1 * np.random.randn(*Y[0].shape)
    return perturbed_Y

def condition_to_check_Y(Y):
    """
    Check the condition to determine if the interpolation set needs improvement.
    
    Parameters:
    Y (list of np.ndarray): Current interpolation set.
    
    Returns:
    bool: True if the set needs improvement, False otherwise.
    """
   
    n = len(Y)
    distances = [np.linalg.norm(Y[i] - Y[j]) for i in range(n) for j in range(i + 1, n)]
    return np.min(distances) < 1e-3

def plot_iteration(Y, xk, xk_plus, delta, k, f):
    """
    Plot the current iteration, including the function, interpolation points, and trust region.
    
    Parameters:
    Y (list of np.ndarray): Current interpolation set.
    xk (np.ndarray): Current point.
    xk_plus (np.ndarray): Trial point.
    delta (float): Trust region radius.
    k (int): Current iteration number.
    f (function): Objective function.
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 3D plot
    ax = fig.add_subplot(121, projection='3d')
    x1_range = np.linspace(-2, 3, 100)
    x2_range = np.linspace(-2, 4, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.array([[f([x1, x2]) for x1 in x1_range] for x2 in x2_range])
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
    
    Y_array = np.array(Y)
    ax.scatter(Y_array[:, 0], Y_array[:, 1], [f(y) for y in Y], c='blue', label='Interpolation Points')
    ax.scatter(*xk, f(xk), c='red', label='Current Point', marker='x')
    ax.scatter(*xk_plus, f(xk_plus), c='green', label='Trial Point', marker='+')
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = delta * np.outer(np.cos(u), np.sin(v)) + xk[0]
    y_sphere = delta * np.outer(np.sin(u), np.sin(v)) + xk[1]
    z_sphere = delta * np.outer(np.ones(np.size(u)), np.cos(v)) + f(xk)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)

    ax.set_title(f"Iteration {k} (3D)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.legend()
    
    # 2D contour plot
    plt.figure(figsize=(6, 6))
    print(1)
    x1_range = np.linspace(-2, 3, 400)
    x2_range = np.linspace(-2, 4, 400)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.array([[f([x1, x2]) for x1 in x1_range] for x2 in x2_range])
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    
    plt.scatter(*zip(*Y), c='blue', label='Interpolation Points')
    plt.scatter(*xk, c='red', label='Current Point', marker='x')
    plt.scatter(*xk_plus, c='green', label='Trial Point', marker='+')
    circle = plt.Circle(xk, delta, color='gray', alpha=0.2, label='Trust Region')
    plt.gca().add_patch(circle)
    plt.title(f"Iteration {k} (2D)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.xlim(-2, 3)
    plt.ylim(-2, 4)
    plt.grid(True)
    
    plt.show()

def derivative_free_optimization(f, Y, delta0, eta, max_iter=100,improvement_threshold=1e-6):
    """
    Perform derivative-free optimization using a model-based method with quadratic interpolation.
    
    Parameters:
    f (function): Objective function.
    Y (list of np.ndarray): Initial interpolation set.
    delta0 (float): Initial trust region radius.
    eta (float): Constant for the acceptance criterion.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    np.ndarray: Optimal point found by the algorithm.
    """
    xk = min(Y, key=f)
    delta = delta0
    k = 0
    f_values = np.array([f(y) for y in Y])
    prev_f_value = np.inf
    start_time = time.time()

    while k < max_iter:
        try:
            coefs = quadratic_model(Y, f_values)
        except ValueError:
            Y = geometry_improving_procedure(Y)
            f_values = np.array([f(y) for y in Y])
            xk = min(Y, key=f)
            coefs = quadratic_model(Y, f_values)
        
        mk = model_value(coefs, xk)
        
        pk = trust_region_subproblem(coefs, xk, delta)
        xk_plus = xk + pk
        
        rho = compute_rho(f, xk, pk, mk, f_values, coefs)
        
        #if k % 10 == 0:
        plot_iteration(Y, xk, xk_plus, delta, k, f)
        
        if rho >= eta:
            Y[np.argmax(f_values)] = xk_plus
            f_values = np.array([f(y) for y in Y])
            delta = max(delta, delta0)
            xk = xk_plus
        else:
            if condition_to_check_Y(Y):
                delta = delta * 0.5
                k += 1
                continue
            
            Y = geometry_improving_procedure(Y)
            f_values = np.array([f(y) for y in Y])
            xk = min(Y, key=f)
            coefs = quadratic_model(Y, f_values)
            rho = compute_rho(f, xk, pk, mk, f_values, coefs)
            
            if rho >= eta:
                xk = xk_plus
        

        if np.abs(f(xk) - prev_f_value) < improvement_threshold:
            break  # If improvement is smaller than threshold, stop
        else:
            prev_f_value = f(xk)  # Update previous objective function value
        
        k += 1  
    
    # Stop the timer
    end_time = time.time()
    running_time = end_time - start_time
    
    return xk, running_time


