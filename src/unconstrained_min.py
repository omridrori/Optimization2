import numpy as np

class LineSearchOptimizer:
    def __init__(self, f, grad, hess=None):
        self.f = f        # Objective function
        self.grad = grad  # Gradient of the objective function
        self.hess = hess  # Hessian of the objective function (optional, for Newton's method)

    def gradient_descent(self, x0, obj_tol, param_tol, max_iter):
        x = x0
        iteration_details = [(0, x.copy(), self.f(x)[0])]
        for i in range(1,max_iter):
            gradient = self.grad(x)
            if np.linalg.norm(gradient) < param_tol:
                break
            step_size = self.line_search(x, -gradient)
            x_new = x - step_size * gradient
            f_new,_,_ = self.f(x_new)
            iteration_details.append((i, x_new.copy(), f_new))
            if abs(self.f(x)[0] - f_new) < obj_tol:
                break
            x = x_new
        return x, self.f(x)[0], i < max_iter-1, iteration_details

    def newton_method(self, x0, obj_tol, param_tol, max_iter):
        x = x0
        iteration_details = [(0, x.copy(), self.f(x)[0])]
        for i in range(1,max_iter):
            gradient = self.grad(x)
            hessian = self.hess(x)
            if np.linalg.cond(hessian.astype("float")) > 1 / np.finfo(float).eps:
                return x, self.f(x)[0], False, iteration_details
            hessian_inv = np.linalg.inv(hessian)

            newton_step = hessian_inv.dot(gradient)
            directional_derivative = newton_step.T @ hessian @ newton_step

            # Check if the directional derivative is less than the specified tolerance
            if directional_derivative < param_tol:
                break
            if np.linalg.norm(newton_step) < param_tol:
                break
            step_size = self.line_search(x, -newton_step)
            x_new = x - step_size * newton_step
            f_new,_,_ = self.f(x_new)
            iteration_details.append((i, x_new.copy(), f_new))
            if abs(self.f(x)[0] - f_new) < obj_tol:
                break
            x = x_new
        return x, self.f(x)[0], i < max_iter, iteration_details

    def line_search(self, x, direction, alpha=1, c1=0.01, beta=0.5, max_iter=50):
        current_f ,_,_= self.f(x)
        current_grad = self.grad(x)
        current_slope = np.dot(current_grad, direction)

        for _ in range(max_iter):
            proposed_x = x + alpha * direction
            proposed_f,_,_ = self.f(proposed_x)
            # Check the Armijo Condition
            if proposed_f  > current_f + c1 * alpha * current_slope:
                alpha *= beta  # Reduce step size if not sufficient decrease
            else:
                return alpha  # Sufficient decrease condition met

        return alpha  # Return alpha even if conditions are not met within max iterations

    def optimize(self, x0, method='gradient_descent', obj_tol=1e-6, param_tol=1e-6, max_iter=100):
        if method == 'gradient_descent':
            return self.gradient_descent(x0, obj_tol, param_tol, max_iter)
        elif method == 'newton':
            return self.newton_method(x0, obj_tol, param_tol, max_iter)
        else:
            raise ValueError("Method not supported")
