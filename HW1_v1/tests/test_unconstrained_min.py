import unittest
import numpy as np
import sys
sys.path.append('/workspaces/Numerical-Optimization-with-Python/')
print('OK')

from HW1_v1.src.unconstrained_min import UnconstrainedMinimizer
from HW1_v1.src.utils import plot_contour, plot_function_values
from HW1_v1.tests.examples import quadratic_example,  rosenbrock_example, linear_example,smoothed_corner_triangles

print('OK')

class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        # Set up examples
        self.examples = {
            "quadratic": quadratic_example,
            "rosenbrock": rosenbrock_example,
            "linear": linear_example,
            "triangle": smoothed_corner_triangles
        }

    def test_minimization(self):
        print("OK_test1")
        for example_name, example_func in self.examples.items():
            print(f"Running tests for {example_name} example...")
            for method in ['gradient_descent', 'newton']:
                minimizer = UnconstrainedMinimizer(method=method)
                print("OK")
                x_opt, f_opt, success = minimizer.minimize(example_func, x0=[1, 1])
                self.assertTrue(success)
                print(f"{method} method converged to {x_opt} with objective value {f_opt}")

                # Plot contour lines with iteration paths
                plot_contour(example_func, example_name, minimizer.iteration_paths, method_names=[method])

                # Plot function values vs. iteration number
                plot_function_values(minimizer.function_values, method_names=[method])

                print()

if __name__ == "__main__":
    print("ok")
    unittest.main()
