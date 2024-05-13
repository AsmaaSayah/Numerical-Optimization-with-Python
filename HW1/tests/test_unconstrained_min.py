import unittest
import numpy as np
import sys
sys.path.append('/workspaces/Numerical-Optimization-with-Python/')
print('OK')

from HW1.src.unconstrained_min import wolfe_backtrack, gradient_descent, newton_descent
from HW1.src.utils import draw_results, plot_contour
from HW1.tests.examples import example_func_quad_1,  example_func_quad_2, example_func_quad_3,example_func_rosenbrock,example_func_linear,example_func_nonquad,func_quad

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
                plot_contour(example_func, example_name)

                # Plot function values vs. iteration number
                plot_function_values(minimizer.function_values, func_name=[method])

                print()



if __name__ == "__main__":
    print("ok")
    unittest.main()




















# test params
max_iter = 1000
step_tol = 1e-8
obj_tol = 1e-12

# Get the requested function to analyze from user

print(1)
print('Choose a function for analysis from the following:')
print('1-circle, 2-ellipse, 3-shifted ellipse, 4-Rosenbrock, 5-linear, 6-nonquad')

function_index = int(input('type a single number between [1, 6]:'))

# Create function dictionary
func_dict = {
    1: ('circle', examples.example_func_quad_1),
    2: ('ellipse', examples.example_func_quad_2),
    3: ('shifted ellipse', examples.example_func_quad_3),
    4: ('Rosenbrock', examples.example_func_rosenbrock),
    5: ('linear', examples.example_func_linear),
    6: ('nonquad', examples.example_func_nonquad)
}
print(1)

if function_index in func_dict:
    func_name, func2min = func_dict[function_index]
    print(f'You chose {function_index}: {func_name}')

    methods = ['gd', 'newton']
    results = {}
    x0 = np.array([8, 6], dtype=np.float64)

    for method in methods:
        if function_index == 5 and method != 'gd':  # Skip for 'newton', 'bfgs' and 'sr1' if Linear
            results[method] = []
        else:
            results[method] = unconstrained_min.unconstrained_minimization(func2min, x0, max_iter, obj_tol, step_tol, method)

    # plot all methods tracks on top of the contour
    print(1)
    utils.plot_contour(func2min, func_name, *results.values())

    print(f'End of {func_name} analysis')

else:
    print(f"You chose {function_index}, it should be an integer number between 1-6. Please rerun and try again.")

print('End of running.')



print("ok")


