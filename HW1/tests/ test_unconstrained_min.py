#test_constrained_min.py
import numpy as np
from src import unconstrained_min
from src import utils
from examples import (
    example_func_quad_1,
    example_func_quad_2,
    example_func_quad_3,
    example_func_rosenbrock,
    example_func_linear,
    example_func_nonquad
)

# test params
max_iter = 1000
step_tol = 1e-8
obj_tol = 1e-12

# Get the requested function to analyze from user

print('Choose a function for analysis from the following:')
print('1-circle, 2-ellipse, 3-shifted ellipse, 4-Rosenbrock, 5-linear, 6-nonquad')

function_index = int(input('type a single number between [1, 6]:'))

# Create function dictionary
func_dict = {
    1: ('circle', example_func_quad_1),
    2: ('ellipse', example_func_quad_2),
    3: ('shifted ellipse', example_func_quad_3),
    4: ('Rosenbrock', example_func_rosenbrock),
    5: ('linear', example_func_linear),
    6: ('nonquad', example_func_nonquad)
}

if function_index in func_dict:
    func_name, func2min = func_dict[function_index]
    print(f'You chose {function_index}: {func_name}')

    methods = ['gd', 'newton', 'bfgs', 'sr1']
    results = {}
    x0 = np.array([8, 6], dtype=np.float64)

    for method in methods:
        if function_index == 5 and method != 'gd':  # Skip for 'newton', 'bfgs' and 'sr1' if Linear
            results[method] = []
        else:
            results[method] = unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # plot all methods tracks on top of the contour
    plot_contour(func2min, func_name, *results.values())

    print(f'End of {func_name} analysis')

else:
    print(f"You chose {function_index}, it should be an integer number between 1-6. Please rerun and try again.")

print('End of running.')



print("ok")

import unittest

from constrained_min import log_barrier  # replace with the actual module name
from examples import func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0

class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        # Call the log_barrier function
        x = log_barrier(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)

        # Check the solution
        np.testing.assert_almost_equal(x, np.array([0.3333, 0.3333, 0.3333]), decimal=4)

if __name__ == '__main__':
    unittest.main()
