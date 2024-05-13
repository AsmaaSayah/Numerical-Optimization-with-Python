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
        - grad: Gradient vector.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        x = x0
        for _ in range(self.max_iter):
            step_length, _ = self.wolfe_backtracking(func, grad, x, grad)
            x_new = x - step_length * grad
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return x

    def newton(self, func, grad, hessian, x0):
        """
        Perform Newton optimization.

        Parameters:
        - func: Objective function.
        - grad: Gradient vector.
        - hessian: Hessian matrix.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        x = x0
        for _ in range(self.max_iter):
            step_length, _ = self.wolfe_backtracking(func, grad, x, grad)
            x_new = x - step_length * np.linalg.inv(hessian) @ grad
            if np.linalg.norm(x_new - x) < self.tol:
                break
            x = x_new
        return x

    def minimize(self, func, grad, hessian, x0):
        """
        Minimize the objective function.

        Parameters:
        - func: Objective function.
        - grad: Gradient vector.
        - hessian: Hessian matrix.
        - x0: Initial guess.

        Returns:
        - x: Optimal solution.
        """
        if self.method == 'gradient_descent':
            x_opt = self.gradient_descent(func, grad, x0)
            # Compute f_opt using the objective function
            f_opt = func(x_opt)[0]
            # Determine success based on convergence criteria
            success = True  # Example, adjust as needed
            return x_opt, f_opt, success
        elif self.method == 'newton':
            x_opt = self.newton(func, grad, hessian, x0)
            # Compute f_opt using the objective function
            f_opt = func(x_opt)[0]
            # Determine success based on convergence criteria
            success = True  # Example, adjust as needed
            return x_opt, f_opt, success
        else:
            raise ValueError("Invalid method. Choose from 'gradient_descent' or 'newton'")

    def wolfe_backtracking(self, func, grad, x, direction):
        """
        Perform backtracking line search with Wolfe conditions.

        Parameters:
        - func: Objective function.
        - grad: Gradient vector.
        - x: Current point.
        - direction: Search direction.

        Returns:
        - step_length: Step length.
        - func_value_next: Objective function value at next point.
        """
        step_length = 1.0
        func_value_current, grad_current = func(x)[0], grad.dot(direction)
        print(grad_current)
        while True:
            next_x = x - step_length * direction
            func_value_next,grad_value_next,hessian_value_next = func(next_x)
            print(grad_value_next,hessian_value_next)
            if (func_value_next > func_value_current + self.c1 * step_length * grad_current or
                    (func_value_next >= func_value_current and step_length > 1e-10)):
                step_length *= self.step_scaling_factor
            else:
                if grad_value_next@direction < self.c2 * grad_current:
                    step_length *= 2
                    break
                elif grad_value_next@direction > -self.c1 * grad_current:
                    step_length *= self.step_scaling_factor
                else:
                    break
        return step_length, func_value_next
