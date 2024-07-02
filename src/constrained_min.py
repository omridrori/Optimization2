import math
import numpy as np


class ConstrainedMinimizer:
    def __init__(self, objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point):
        self.objective_fn = objective_fn
        self.inequality_constraints = inequality_constraints
        self.equality_matrix = equality_matrix
        self.equality_rhs = equality_rhs
        self.initial_point = initial_point.copy()

        self.barrier_param = 1
        self.barrier_multiplier =1000
        self.tolerance_obj = 10e-10
        self.stopping_criteria = 10e-10
        self.duality_gaps = []
        self.total_iterations = 0


    def minimize(self):
        curr_point = self.initial_point
        obj_val, grad_val, hess_val = self.objective_fn(curr_point, True)
        barrier_obj_val, barrier_grad_val, barrier_hess_val = self.barrier_function(curr_point)

        inner_points_history = [curr_point.copy()]
        inner_obj_history = [obj_val]
        outer_points_history = [curr_point.copy()]
        outer_obj_history = [obj_val]

        obj_val, grad_val, hess_val = self.combine_objectives(self.barrier_param, obj_val, grad_val, hess_val,
                                                              barrier_obj_val, barrier_grad_val, barrier_hess_val)
        iteration_outer = 0
        while True and iteration_outer < 30:
            kkt_mat, kkt_vec = self.kkt_system(grad_val, hess_val)

            prev_point = curr_point
            prev_obj_val = obj_val

            iteration = 0
            while True and iteration < 150:
                if iteration != 0 and self.is_converged(curr_point, prev_point):
                    break

                search_dir = np.linalg.solve(kkt_mat, kkt_vec)[:curr_point.shape[0]]

                # Compute lambda for the new constraints
                _lambda = np.matmul(search_dir.transpose(), np.matmul(hess_val, search_dir)) ** 0.5

                if 0.5 * (_lambda ** 2) < self.tolerance_obj:
                    break

                if iteration != 0 and self.is_small_reduction(prev_obj_val, obj_val):
                    break

                step_size = self.line_search(search_dir, curr_point)

                prev_point = curr_point
                prev_obj_val = obj_val

                curr_point, obj_val, grad_val, hess_val, barrier_obj_val, barrier_grad_val, barrier_hess_val = self.update_point(
                    curr_point, step_size, search_dir)

                inner_points_history.append(curr_point)
                inner_obj_history.append(obj_val)

                obj_val, grad_val, hess_val = self.combine_objectives(self.barrier_param, obj_val, grad_val, hess_val,
                                                                      barrier_obj_val, barrier_grad_val,
                                                                      barrier_hess_val)

                iteration += 1

            outer_points_history.append(curr_point)
            outer_obj_history.append((obj_val - barrier_obj_val) / self.barrier_param)

            if self.should_stop():
                break

            self.barrier_param *= self.barrier_multiplier
            iteration_outer+=1
            self.duality_gaps.append(len(self.inequality_constraints) / self.barrier_param)
            self.total_iterations += iteration
        return inner_points_history, inner_obj_history, outer_points_history, outer_obj_history

    def kkt_system(self, gradient, hessian):
        if self.equality_matrix.size > 0:
            upper_block = np.concatenate([hessian, self.equality_matrix.T], axis=1)
            lower_block = np.concatenate(
                [self.equality_matrix, np.zeros((self.equality_matrix.shape[0], self.equality_matrix.shape[0]))],
                axis=1)
            kkt_matrix = np.concatenate([upper_block, lower_block], axis=0)
        else:
            kkt_matrix = hessian

        kkt_vector = np.concatenate([-gradient, np.zeros(kkt_matrix.shape[0] - len(gradient))])

        return kkt_matrix, kkt_vector

    def barrier_function(self, point):
        barrier_val = 0.0
        barrier_grad = np.zeros_like(point)
        barrier_hess = np.zeros((point.shape[0], point.shape[0]))

        for constraint in self.inequality_constraints:
            constr_val, constr_grad, constr_hess = constraint(point, True)

            barrier_val -= math.log(-constr_val)
            grad_contrib = constr_grad / constr_val
            barrier_grad -= grad_contrib
            barrier_hess -= (constr_hess * constr_val - np.outer(grad_contrib, grad_contrib)) / constr_val ** 2

        return barrier_val, barrier_grad, barrier_hess

    def line_search(self, direction, current_point):
        wolfe_const = 0.01
        backtrack_const = 0.5
        step = 1

        def wolfe_conditions(step_size):

            if not all([constraint(current_point + step_size * direction)[0] < 0 for constraint in self.inequality_constraints]):
                return False

            sufficient_decrease = self.objective_fn(current_point + step_size * direction)[0] <= \
                                  self.objective_fn(current_point)[0] + wolfe_const * step_size * (
                                              self.objective_fn(current_point)[1] @ direction)
            return sufficient_decrease

        while not wolfe_conditions(step):
            step *= backtrack_const

        return step

    @staticmethod
    def combine_objectives(barrier_param, obj_val, grad_val, hess_val, barrier_obj_val, barrier_grad_val,
                           barrier_hess_val):
        obj_val = barrier_param * obj_val + barrier_obj_val
        grad_val = barrier_param * grad_val + barrier_grad_val
        hess_val = barrier_param * hess_val + barrier_hess_val
        return obj_val, grad_val, hess_val

    @staticmethod
    def is_converged(curr_point, prev_point):
        return sum(abs(curr_point - prev_point)) < 10e-8

    def is_small_step(self, search_dir, hess_val):
        return 0.5 * ((search_dir.T @ (hess_val @ search_dir)) ** 0.5) ** 2 < self.tolerance_obj

    def is_small_reduction(self, prev_obj_val, obj_val):
        return (prev_obj_val - obj_val) < self.tolerance_obj

    def update_point(self, curr_point, step_size, search_dir):
        new_point = curr_point + step_size * search_dir
        obj_val, grad_val, hess_val = self.objective_fn(new_point, True)
        barrier_obj_val, barrier_grad_val, barrier_hess_val = self.barrier_function(new_point)
        return new_point, obj_val, grad_val, hess_val, barrier_obj_val, barrier_grad_val, barrier_hess_val

    def should_stop(self):
        return len(self.inequality_constraints) / self.barrier_param < self.stopping_criteria


# def interior_pt(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point):
#     minimizer = ConstrainedMinimizer(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point)
#     return minimizer.minimize()

def interior_pt(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point,minimizer=None):
    if minimizer is None:
        minimizer = ConstrainedMinimizer(objective_fn, inequality_constraints, equality_matrix, equality_rhs,
                                         initial_point)
    inner_points_history, inner_obj_history, outer_points_history, outer_obj_history = minimizer.minimize()
    return inner_points_history, inner_obj_history, outer_points_history, outer_obj_history, minimizer.duality_gaps

def run_interior_point_for_multiple_mu(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point, mu_values=[1,2,3,4,5,10,15,20,100,1000]):
    results = []
    for mu in mu_values:
        minimizer = ConstrainedMinimizer(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point)
        minimizer.barrier_multiplier = mu
        inner_points_history, inner_obj_history, outer_points_history, outer_obj_history, duality_gaps = interior_pt(objective_fn, inequality_constraints, equality_matrix, equality_rhs, initial_point,minimizer)
        total_iterations = len(inner_points_history)
        results.append((mu, duality_gaps, total_iterations))
    return results