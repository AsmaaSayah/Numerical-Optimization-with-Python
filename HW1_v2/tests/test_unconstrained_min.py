import numpy as np
import unittest
import sys
sys.path.append('/workspaces/Numerical-Optimization-with-Python/')
from HW1_v2.src.unconstrained_min import unconstrained_minimization
from HW1_v2.src.utils import plot_contour,plot_function_values
from HW1_v2.tests.examples import (
    example_func_quad_1,
    example_func_quad_2,
    example_func_quad_3,
    example_func_rosenbrock,
    example_func_linear,
    example_func_nonquad
)

# test params
step_tol = 1e-8
obj_tol = 1e-12

class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        # Create function dictionary
        self.func_dict = {
            1: ('circle (contour lines are circles)', example_func_quad_1),
            2: ('ellipse (contour lines are axis aligned ellipses)', example_func_quad_2),
            3: ('shifted ellipse (contour lines are rotated ellipses)', example_func_quad_3),
            4: ('Rosenbrock', example_func_rosenbrock),
            5: ('linear', example_func_linear),
            6: ('nonquad (smoothed corner triangles)', example_func_nonquad)
        }

    def tearDown(self):
        print('End of running.')

    def test_minimization(self):
        for function_index in range(1, 7):
            func_name, func2min = self.func_dict[function_index]

            methods = ['gd', 'newton']
            results = {}
            if function_index == 4:
                x0 = np.array([-1, 2], dtype=np.float64)  # For Rosenbrock example
                max_iter = 10000  # For Gradient Descent with Rosenbrock example
            else:
                x0 = np.array([1, 1], dtype=np.float64)
                max_iter = 100

            for method in methods:
                if function_index == 5 and method != 'gd':
                    results[method] = []
                else:
                    results[method] = unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

                # You may want to perform assertions on the results here
            print(func_name)
            # Plot contour lines with iteration paths
            plot_contour(func2min, func_name, *results.values())
            plot_function_values(func_name, *results.values())

            print(f'End of {func_name} analysis')


if __name__ == '__main__':
    unittest.main()
