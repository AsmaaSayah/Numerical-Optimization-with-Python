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
        initial_points = {
            "quadratic": np.array([1, 1]),
            "rosenbrock": np.array([-1, 2]),
            "linear": np.array([1, 1]),
            "triangle": np.array([1, 1])
        }
        for example_name, example_func in self.examples.items():
            print(f"Running tests for {example_name} example...")
            for method in ['gradient_descent', 'newton']:
                minimizer = UnconstrainedMinimizer(method=method)
                if example_name == "rosenbrock":
                    # Linear function doesn't require Hessian
                    x0_init=initial_points[example_name]
                    f, grad, h = example_func(x0_init)

                else:
                    # Use the computed grad and hessian from the example function
                    x0_init=initial_points[example_name]
                    f, grad, h = example_func(x0_init)
                print("test_start_here")
                print(x0_init)
                print(f,grad,h)
                x_opt, f_opt, success = minimizer.minimize(example_func, grad, h, x0=x0_init)
                self.assertTrue(success)
                print(f"{method} method converged to {x_opt} with objective value {f_opt}")

                # Plot contour lines with iteration paths
                plot_contour(example_func, paths=minimizer.iteration_paths, labels=[method])

                # Plot function values vs. iteration number
                plot_function_values(minimizer.function_values)


                print()



if __name__ == "__main__":
    print("ok")
    unittest.main()
