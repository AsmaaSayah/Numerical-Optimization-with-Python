import numpy as np
import unittest
import sys
sys.path.append('/workspaces/Numerical-Optimization-with-Python/')
from HW1_v2.src.constrained_min import interior_pt
from HW1_v2.src.utils import plot_results_linear, plot_results_quadratic
from examples import test_qp, test_lp

class TestConstrainedMinimization(unittest.TestCase):
    def test_quadratic_function(self):
        func_name = 'quadratic'
        backtrack_flag = True
        func_min = test_qp
        start_x = np.array([0.1, 0.2, 0.7], dtype=np.float64)
        newton_outcome = interior_pt(func_min, start_x, backtrack=backtrack_flag, m=4, t=1.0, miu=10, eps_barrier=1e-5,
                                     eps_newton=1e-5)

        # Plot results
        x_trajectory = newton_outcome[2]
        f_trajectory = newton_outcome[3]
        x_limits = np.array([-2, 2])
        y_limits = np.array([-2, 2])
        z_limits = np.array([-2, 2])
        plot_results_quadratic(func_min, x_trajectory, f_trajectory, x_limits, y_limits, z_limits, func_name)

    def test_linear_function(self):
        func_name = 'linear'
        backtrack_flag = False
        func_min = test_lp
        start_x = np.array([0.5, 0.75], dtype=np.float64)
        newton_outcome = interior_pt(func_min, start_x, backtrack=backtrack_flag, m=4, t=1.0, miu=10, eps_barrier=1e-5,
                                     eps_newton=1e-5)

        # Plot results
        x_trajectory = newton_outcome[2]
        f_trajectory = newton_outcome[3]
        x_limits = np.array([-1, 3])
        y_limits = np.array([-1, 3])
        plot_results_linear(func_min, x_trajectory, f_trajectory, x_limits, y_limits, func_name)

if __name__ == '__main__':
    unittest.main()