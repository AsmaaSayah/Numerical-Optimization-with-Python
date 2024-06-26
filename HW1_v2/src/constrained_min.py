import numpy as np
from scipy.optimize import minimize
from autograd import grad
from autograd import hessian
import autograd.numpy as anp

t = 1


class LogBarrier:
    def __init__(self,
                 func,
                 ineq_constraints,
                 eq_constraints_mat,
                 eq_constraints_rhs,
                 x0):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.x0 = x0

    def line_search(self, func, grad, xk, pk, c1=0.01, alpha_init=1.0):
        alpha = alpha_init
        while True:
            func_xk = func(xk)
            func_xk_alpha_pk = func(xk + alpha * pk)
            grad_xk_dot_pk = np.dot(grad.T, pk)

            if func_xk_alpha_pk > func_xk + c1 * alpha * grad_xk_dot_pk:
                alpha *= 0.5
            else:
                return alpha

    def phi(self, x):
        phi_value = 0
        for constraint in self.ineq_constraints:
            if -constraint(x) <= 0:  # if the constraint is violated
                print("ERROR negative log")
                return np.inf  # returning a large value indicating infeasible solution
            else:
                phi_value -= anp.log(-constraint(x))  # accumulate the log barrier for each constraint
        return phi_value

    def interior_pt(self, test_name, mu=10, eps=1e-5, max_iter=1000):
        global t
        n = len(self.x0)
        m = len(self.ineq_constraints)

        def objective(x):
            return t * self.func(x) + self.phi(x)

        path = []
        objectives = []

        fs = [self.func(self.x0)]

        x = self.x0
        k = 0
        while m / t > eps:
            path.append(np.copy(x))
            objectives.append(objective(x))

            print(f'Iteration: {k} \t x = {x}, f(x) = {fs[k]:.4f}, gap = {m / t:.4f}')

            i = 0
            p_nt = np.array([[1], [1]])  # Initialize to any non-zero value
            while np.linalg.norm(p_nt) > eps and i < max_iter:
                grad_func, grad_phi, gx_func, gx_phi, gx = self.gradient(x)
                hessian_func, hessian_phi, hx_func, hx_phi, hx = self.hessian(x)

                p_nt = self.newton_step(hx, gx)

                alpha = self.line_search(objective, gx, x, p_nt)
                x = x + alpha * p_nt

                i += 1
                print(f'    Iteration2: {i} \t x = {x}, p_nt={p_nt}, np.linalg.norm(p_nt) > eps={np.linalg.norm(p_nt) > eps}')

            fs.append(self.func(x))
            k += 1
            t *= mu

        return x, path, objectives

    def newton_step(self, H, g, reg_term=1e-6):
        A = self.eq_constraints_mat
        b = self.eq_constraints_rhs
        # The shape of H, which is the Hessian matrix
        n = H.shape[0]

        # Regularize the Hessian to avoid singularity
        H += np.eye(n) * reg_term

        # If there are no equality constraints, solve the system of equations to find d
        if A is None:
            p_nt = np.linalg.solve(H, -g)
            return p_nt
        else:
            # If there are equality constraints, construct the KKT system
            m = A.shape[0]  # Number of constraints
            # Reshaping A for proper concatenation
            A = np.reshape(A, (1, -1))
            AT = np.reshape(A.T, (-1, 1))

            # Zero matrix and zero vector
            zeros_m_m = np.zeros((len(b),len(b)))
            zeros_m = np.zeros((len(b),len(b)))

            KKT_mat = np.vstack([np.hstack((H, AT)), np.hstack((A, zeros_m_m))])
            KKT_rhs = np.hstack((np.squeeze(-g), zeros_m.flatten()))


            # Solve the KKT system
            sol = np.linalg.solve(KKT_mat, KKT_rhs)

            # The solution includes the step direction 'd' and the Lagrange multipliers 'lambda_'
            p_nt = sol[:n]
            lambda_ = sol[n:]

            return p_nt

    def gradient(self, x):
        grad_func = grad(self.func)
        grad_phi = grad(self.phi)
        gx_func = t * grad_func(x)
        gx_phi = grad_phi(x)
        gx = gx_func + gx_phi
        print(gx)
        return grad_func, grad_phi, gx_func, gx_phi, gx

    def hessian(self, x):
        hessian_func = hessian(self.func)
        hessian_phi = hessian(self.phi)
        hx_func = t * hessian_func(x)
        hx_phi = hessian_phi(x)
        hx = hx_func + hx_phi

        return hessian_func, hessian_phi, hx_func, hx_phi, hx
