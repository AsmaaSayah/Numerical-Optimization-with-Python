import numpy as np


def wolfe_backtrack(func2min, xi, step_direct, t, max_iter=100):
    """
    Wolfe backtracking condition.

    Input:
    - func2min: objective function to minimize
    - x: current position.
    - p: step direction

    Output:
    - alpha: i.e., step length

    """
    # Init
    alpha = 1.0
    c_wolfe_1 = 1e-3
    alpha_factor = 0.5
    p = step_direct
    alpha_ok = False

    # current position f(x) & g(x)
    func_x, grad_x, h_NA = func2min(xi, t)
    # <g(x),p>
    grad_proj_step = np.dot(grad_x, p)

    for i in range(max_iter):
        # Evaluate next position f(x+ap)
        x_next = xi + alpha * p
        func_x_next, g_NA, h_NA = func2min(x_next, t)

        # Check the sufficient decrease condition (i.e., Wolfe #1)
        if func_x_next < func_x + alpha*c_wolfe_1*grad_proj_step:
            alpha_ok = True
            return alpha, alpha_ok

        # Modify alpha and re-check for a better fit of the step length
        alpha *= alpha_factor

    return alpha, alpha_ok


def interior_pt(func2min, x0, m, t, miu=10, eps_barrier=1e-5, eps_newton=1e-5, max_iter=1000, backtrack=False):

    """
    Barrier method algorithm.

    Input:
    - func2min: objective function to minimize
    - x0: initial position.
    - m: number of constraints
    - t: user's parameter
    - miu: t slope
    - eps_barrier: barrier stop criteria
    - eps_newton: newton step stop criteria
    - max_iter: inner loop max number of iterations

    Output: track records
    - x_track, f_track & barrier_factor

    """

    xi = np.copy(x0)  # store initial value
    # Init path records
    dim_x = len(xi)
    x_track = np.array(np.zeros((max_iter, dim_x)), dtype=np.float64)
    f_track = np.array(np.zeros(max_iter), dtype=np.float64)

    #x_track = []  # keep track on x
    #f_track = []  # keep track on f(x)
    outer_iter = 0  # number of iterations of outer loop

    # loop until stopping criterion is met
    while m/t > eps_barrier:
        # Get f(x), g(x) & H(x)
        func_x, grad_x, hessian_x = func2min(xi, t)

        # find Newton direction using equation solver:
        newton_step = -np.linalg.solve(hessian_x, grad_x)

        # Keep initial positions
        if outer_iter == 0:
            # append to track records
            x_track[outer_iter, :] = xi
            f_track[outer_iter] = func_x
            print('Barrier init: x = {}, f(x) = {}, m/t = {}'.format(xi, func_x, m / t))

        # centering step: Newton Algorithm
        inner_iter = 0

        while inner_iter < max_iter:
            prev_func_x = np.copy(func_x)

            # find step length using wolfe backtrack condition
            if backtrack == False:
                alpha, alpha_ok = 0.05, True
            else:
                alpha, alpha_ok = wolfe_backtrack(func2min, xi, newton_step, t)

            if alpha_ok == False:
                break

            # take 1 step towards opposite of current gradient
            xi += alpha * newton_step

            # set the next step direction
            func_x, grad_x, hessian_x = func2min(xi, t)

            df = prev_func_x - func_x
            if df <= eps_newton:
                #print("Newton termination: small df =", df)
                break

            newton_step = -np.linalg.solve(hessian_x, grad_x)

            inner_iter += 1

        # print result
        print('Barrier iteration #{}: x = {}, f(x) = {}, m/t = {}'.format(outer_iter, xi, func_x, m/t))

        # update parameter t
        t *= miu
        # Increment outer loop iteration
        outer_iter += 1

        # append to track records
        x_track[outer_iter, :] = xi
        f_track[outer_iter] = func_x

    # Output results
    x_track = x_track[:(outer_iter+1), :]
    f_track = f_track[:(outer_iter+1)]
    final_x = xi
    final_fx = func_x

    return final_x, final_fx, x_track, f_track