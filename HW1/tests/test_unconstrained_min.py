#test_constrained_min.py
import numpy as np
import unconstrained_min
import utils
import examples

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


