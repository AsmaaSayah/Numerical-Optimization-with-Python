# src/unconstrained_min.py

import numpy as np

class UnconstrainedMinimizer:
    def __init__(self, method='gradient_descent', alpha=0.01, tol=1e-8, max_iter=100,
                 c1=0.01, c2=0.5, step_scaling_factor=0.5):
        """
        Initialize the UnconstrainedMinimizer object.

        Parameters:
        - method: Optimization method, either 'gradient_descent' or 'newton'.
        - alpha: Step size for gradient descent.
        - tol: Tolerance for termination.
        - max_iter: Maximum number of iterations.
        - c1: Wolfe condition parameter for sufficient decrease.
        - c2: Wolfe condition parameter for curvature.
        - step_scaling_factor: Scaling factor for step length.
        """
        print("OK_unconstrained_min")
        self.method = method
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.step_scaling_factor = step_scaling_factor

    def gradient_descent(self, func, grad, x0):
        """
        Perform gradient descent optimization.

        Parameters:
        - func: Objective function.
        - grad: Gradient function.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        x = x0
        for _ in range(self.max_iter):
            gradient = grad(x)
            step_length, _ = self.wolfe_backtracking(func, grad, x, gradient)
            x_new = x - step_length * gradient
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return x

    def newton(self, func, grad, hessian, x0):
        """
        Perform Newton optimization.

        Parameters:
        - func: Objective function.
        - grad: Gradient function.
        - hessian: Hessian function.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        x = x0
        for _ in range(self.max_iter):
            gradient = grad(x)
            hess = hessian(x)
            step_length, _ = self.wolfe_backtracking(func, grad, x, gradient)
            x_new = x - step_length * np.linalg.inv(hess) @ gradient
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return x

    def minimize(self, func, grad, hessian, x0):
        """
        Minimize the objective function.

        Parameters:
        - func: Objective function.
        - grad: Gradient function.
        - hessian: Hessian function.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        print("OK_unconstrained_min")
        if self.method == 'gradient_descent':
            return self.gradient_descent(func, grad, x0)
        elif self.method == 'newton':
            return self.newton(func, grad, hessian, x0)
        else:
            raise ValueError("Invalid method. Choose from 'gradient_descent' or 'newton'")

    def wolfe_backtracking(self, func, grad, x, direction):
        """
        Perform backtracking line search with Wolfe conditions.

        Parameters:
        - func: Objective function.
        - grad: Gradient function.
        - x: Current point.
        - direction: Search direction.

        Returns:
        - step_length: Step length.
        - func_value_next: Objective function value at next point.
        """
        step_length = 1.0
        func_value_current, grad_current = func(x), grad(x).dot(direction)
        while True:
            next_x = x - step_length * direction
            func_value_next = func(next_x)
            if (func_value_next > func_value_current + self.c1 * step_length * grad_current or
                    (func(next_x) >= func(x) and step_length > 1e-10)):
                step_length *= self.step_scaling_factor
            else:
                if grad(next_x).dot(direction) < self.c2 * grad_current:
                    step_length *= 2
                    break
                elif grad(next_x).dot(direction) > -self.c1 * grad_current:
                    step_length *= self.step_scaling_factor
                else:
                    break
        return step_length, func_value_next
