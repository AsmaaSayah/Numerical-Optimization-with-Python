#unconstrained_min.py
import numpy as np

def wolfe_backtrack(func_to_minimize, current_position, step_direction):
    """
    This function implements the Wolfe backtracking condition.

    Parameters:
    - func_to_minimize: The objective function to be minimized.
    - current_position: The current location in the function space.
    - step_direction: The direction of the step.

    Returns:
    - step_length: The length of the step.
    - condition_met: Whether the condition has been met.

    """
    # Initialization
    # Ensure 0 < c1 < c2 < 1
    step_length = 1.0
    wolfe_c1 = 0.0001
    wolfe_c2 = 0.9  # Not needed in this case
    step_length_scaling_factor = 0.25
    compute_hessian = False
    max_iterations = 1000
    condition_met = True

    # Compute function value and gradient at current position
    current_direction = step_direction
    func_value_current, grad_current, _ = func_to_minimize(current_position, compute_hessian)
    projected_grad_step = np.dot(grad_current, current_direction)

    # Compute function value and gradient at next position
    next_position = current_position - step_length * current_direction
    func_value_next, grad_next, _ = func_to_minimize(next_position, compute_hessian)

    # Wolfe condition thresholds
    grad_proj_step_diff = step_length * wolfe_c1 * projected_grad_step

    iteration_counter = 0
    while ((func_value_next > func_value_current - grad_proj_step_diff) and
           (iteration_counter <= max_iterations)):
        # Update iteration counter and step length
        iteration_counter += 1
        step_length *= step_length_scaling_factor

        # Compute function value and gradient at new next position
        next_position = current_position - step_length * current_direction
        func_value_next, grad_next, _ = func_to_minimize(next_position, compute_hessian)

        grad_proj_step_diff = step_length * wolfe_c1 * projected_grad_step

    print(" Backtrack: step_length = {}, func_value_current = {}, func_value_next = {}, grad_proj_step_diff = {} :"
          .format(step_length, func_value_current, func_value_next, grad_proj_step_diff))

    if iteration_counter > max_iterations:
        condition_met = False

    return step_length, condition_met



def wolfe_backtracking_condition(func_to_minimize, current_position, step_direction):
    """
    This function implements the Wolfe backtracking condition.

    Parameters:
    - func_to_minimize: The objective function to be minimized.
    - current_position: The current location in the function space.
    - step_direction: The direction of the step.

    Returns:
    - step_length: The length of the step.
    - condition_met: Whether the condition has been met.

    """
    # Initialization
    # Ensure 0 < c1 < c2 < 1
    step_length = 1.0
    wolfe_c1 = 0.0001
    wolfe_c2 = 0.9  # Not needed in this case
    step_length_scaling_factor = 0.25
    compute_hessian = False
    max_iterations = 1000
    condition_met = True

    # Compute function value and gradient at current position
    current_direction = step_direction
    func_value_current, grad_current, _ = func_to_minimize(current_position, compute_hessian)
    projected_grad_step = np.dot(grad_current, current_direction)

    # Compute function value and gradient at next position
    next_position = current_position - step_length * current_direction
    func_value_next, grad_next, _ = func_to_minimize(next_position, compute_hessian)

    # Wolfe condition thresholds
    grad_proj_step_diff = step_length * wolfe_c1 * projected_grad_step

    iteration_counter = 0
    while ((func_value_next > func_value_current - grad_proj_step_diff) and
           (iteration_counter <= max_iterations)):
        # Update iteration counter and step length
        iteration_counter += 1
        step_length *= step_length_scaling_factor

        # Compute function value and gradient at new next position
        next_position = current_position - step_length * current_direction
        func_value_next, grad_next, _ = func_to_minimize(next_position, compute_hessian)

        grad_proj_step_diff = step_length * wolfe_c1 * projected_grad_step

    print(" Backtrack: step_length = {}, func_value_current = {}, func_value_next = {}, grad_proj_step_diff = {} :"
          .format(step_length, func_value_current, func_value_next, grad_proj_step_diff))

    if iteration_counter > max_iterations:
        condition_met = False

    return step_length, condition_met


def gradient_descent(func_to_minimize, initial_position, max_iterations, tolerance_objective,
                              tolerance_parameters):
    """
    This function implements the gradient descent optimization algorithm.

    Parameters:
    - func_to_minimize: The objective function to be minimized.
    - initial_position: The starting point in the function space.
    - max_iterations: The maximum number of iterations.
    - tolerance_objective: The tolerance for changes in the objective function value.
    - tolerance_parameters: The tolerance for changes in the parameter values.

    Returns:
    - final_position: The final position in the function space.
    - final_objective_value: The final objective function value.
    - positions_track: The positions across iterations.
    - objective_values_track: The objective function values across iterations.
    - optimization_success: Flag indicating whether the optimization was successful.

    """
    # Initialize success flag
    optimization_success = True

    # No need for Hessian in gradient descent
    compute_hessian = False

    # Set the initial position
    current_position = initial_position

    # Initialize records of positions and objective function values
    position_dim = len(current_position)
    positions_track = np.zeros((max_iterations, position_dim), dtype=np.float64)
    objective_values_track = np.zeros(max_iterations, dtype=np.float64)

    for iteration in range(max_iterations):
        # Compute function value and gradient at current position
        objective_value, gradient, _ = func_to_minimize(current_position, compute_hessian)

        # Record position and objective function value
        positions_track[iteration, :] = current_position
        objective_values_track[iteration] = objective_value

        # Print current iteration status
        print("Gradient Descent: iteration = {},  position = {}, objective function value = {} :".format(iteration + 1,
                                                                                                         current_position,
                                                                                                         objective_value))

        # Check termination conditions
        if iteration >= 1:
            position_change = positions_track[iteration - 1] - positions_track[iteration]
            if np.all(position_change <= tolerance_parameters):
                print("Gradient Descent termination: small change in position =", position_change)
                break

            objective_change = objective_values_track[iteration - 1] - objective_values_track[iteration]
            if np.all(objective_change <= tolerance_objective):
                print("Gradient Descent termination: small change in objective function value =", objective_change)
                break

        # Compute step length using Wolfe backtracking condition
        step_direction = gradient
        step_length, step_length_condition_met = wolfe_backtracking_condition(func_to_minimize, current_position,
                                                                              step_direction)

        if not step_length_condition_met:
            break

        # Update position
        current_position -= step_length * gradient

    if (iteration >= max_iterations - 1) or (not step_length_condition_met):
        optimization_success = False

    # Print final status
    print(
        "Gradient Descent final: iteration = {},  position = {}, objective function value = {}, optimization success = {}:"
        .format(iteration + 1, current_position, objective_value, optimization_success))

    # Trim recorded positions and objective function values
    positions_track = positions_track[:(iteration + 1), :]
    objective_values_track = objective_values_track[:(iteration + 1)]

    final_position = current_position
    final_objective_value = objective_value

    return final_position, final_objective_value, positions_track, objective_values_track, optimization_success


def newton_descent(func2min, initial_position, max_iterations, objective_tolerance, parameter_tolerance):
    """
    Newton's method for optimization.

    For detailed description of inputs and outputs, please refer to the documentation of the gradient descent function.
    """
    # Initialization of success flag
    success_flag = True

    # Enable the computation of Hessian
    compute_hessian = True

    # Set the initial position
    current_position = initial_position

    # Initialization of tracking arrays
    position_dimension = len(current_position)
    position_track = np.zeros((max_iterations, position_dimension), dtype=np.float64)
    objective_track = np.zeros(max_iterations, dtype=np.float64)

    for iteration in range(max_iterations):
        # Compute function value, gradient, and Hessian at current position
        func_value, gradient, hessian = func2min(current_position, compute_hessian)

        # Record current position and function value
        position_track[iteration, :] = current_position
        objective_track[iteration] = func_value

        # Print the status of the current iteration
        print(f"Newton: Iteration = {iteration + 1},  Position = {current_position}, Function Value = {func_value}")

        # Check termination conditions
        if iteration >= 1:
            change_in_position = position_track[iteration - 1] - position_track[iteration]
            if np.all(change_in_position <= parameter_tolerance):
                print(f"Newton Termination: Small change in position = {change_in_position}")
                break

            change_in_objective = objective_track[iteration - 1] - objective_track[iteration]
            if np.all(change_in_objective <= objective_tolerance):
                print(f"Newton Termination: Small change in objective = {change_in_objective}")
                break

        # Compute Newton's step
        newton_step = np.linalg.solve(hessian, gradient)

        # Determine step length using Wolfe's backtrack condition
        step_length, step_acceptable = wolfe_backtrack(func2min, current_position, newton_step)

        if not step_acceptable:
            break

        # Update position using Newton's step
        current_position -= step_length * newton_step

    # Set failure flag if maximum iterations reached or if step is not acceptable
    if (iteration >= max_iterations - 1) or (not step_acceptable):
        success_flag = False

    # Print the final status
    print(
        f"Newton Final: Iteration = {iteration + 1},  Position = {current_position}, Function Value = {func_value}, Success = {success_flag}")

    # Truncate tracking arrays to actual size
    position_track = position_track[:(iteration + 1), :]
    objective_track = objective_track[:(iteration + 1)]
    final_position = current_position
    final_value = func_value

    return final_position, final_value, position_track, objective_track, success_flag


def unconstrained_minimization(func2min, x0, max_iter, obj_tol, param_tol, method):
    """
    Input:
    - method_func: function argument to support various minimization methods.
                   Supported methods are the following:
                   Gradient descent and Newton
    - f (func2min): is the function minimized
    - x0: starting point.
    - max_iter: maximum allowed number of iterations.
    - obj_tol: is the numeric tolerance for successful termination in terms of small enough
    change in objective function values, between two consecutive iterations (𝑓(𝑥𝑖+1) and𝑓(𝑥𝑖)), 
    or in the Newton Decrement based approximation of the objective decrease. 
    - param_tol: is the numeric tolerance for successful termination in terms of small enough
    distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).

    Returns:
    - final location
    - final objective value
    - success/failure Boolean flag
    """
    print('The chosen method is =', method)

    if method == 'gd':
        final_x, final_fx, x_track, f_track, success_flag = gradient_descent(func2min, x0, max_iter=10000, obj_tol, param_tol)
    elif method == 'newton':
        final_x, final_fx, x_track, f_track, success_flag = newton_descent(func2min, x0, max_iter, obj_tol, param_tol)
    else:
        final_x = None
        final_fx = None
        x_track = None
        f_track = None
        success_flag = False
        print(
            'The chosen method does not fit. Please make you use on of the following strings: gd, newton, bfgs or sr1')

    return (final_x, final_fx, x_track, f_track, success_flag, method)



print("ok")
    
    
    
    
    
    
    
    
    