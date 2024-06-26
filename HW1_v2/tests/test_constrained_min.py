import unittest
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/workspaces/Numerical-Optimization-with-Python/')
from HW1_v2.src.constrained_min import LogBarrier
from examples import *
from HW1_v2.src.utils import *


class TestConstrainedMin(unittest.TestCase):
    def run_check(self, test_name, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        log_barrier = LogBarrier(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        solution, path, objectives = log_barrier.interior_pt(test_name)

        final_objective = func(solution)
        final_constraints = [constr(solution) for constr in ineq_constraints]

        # The final candidate
        print(f"The end point is = {solution}")
        # Objective and constraint values at the final candidate
        print(f"The final objective value is = {final_objective}")
        print(f"The final constraint values are = {final_constraints}")
        # Plot the feasible region and the path taken by the algorithm.
        plot_path(np.array(path))
        # Plot the objective value vs. outer iteration number
        plot_obj_vs_iter(objectives)
        plot_feasible_region_3d(path)
        plot_path_3d(path)
        #plot_feasible_region_2d()
        #plot_path_2d(path)

    def test_qp(self):
        quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = example_quadratic()
        self.run_check('qp', quadratic, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        
### Run each test seperatly:

    #def test_lp(self):
    #    linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0 = example_linear()
    #    self.run_check('lp', linear, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)


if __name__ == "__main__":
    unittest.main()



