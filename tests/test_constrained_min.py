import unittest
import numpy as np
from src.constrained_min import interior_pt, run_interior_point_for_multiple_mu
from src.utils import plot_iterations, plot_feasible_regions_3d, plot_feasible_set_2d, plot_duality_gap_vs_iterations
from tests.examples import qp_function, qp_inq1, qp_inq2, qp_inq3, lp_func, lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4, \
    qp_equality


class TestInteriorPointOptimization(unittest.TestCase):
    def test_quadratic_program(self):

        # Set up the problem constraints
        inequality_constraints = [qp_inq1, qp_inq2, qp_inq3]
        equality_matrix = np.array([[1, 1, 1]]).reshape(1, -1)
        equality_rhs = np.array([1])
        initial_guess = np.array([0.1, 0.2, 0.7], dtype=np.float64)




        # Execute the solver
        history_inner_x, history_inner_obj, _, history_outer_obj,_ = interior_pt(qp_function, inequality_constraints,
                                                                               equality_matrix, equality_rhs, initial_guess)

        # Display the results
        print(f"Final solution: {history_inner_x[-1]}")
        print(f"Objective value at the final solution: {qp_function(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 1 value at the final solution: {qp_inq1(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 2 value at the final solution: {qp_inq2(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 3 value at the final solution: {qp_inq3(history_inner_x[-1])[0]:.3f}")
        print(f"Equality constraint 4 value at the final solution: {qp_equality(history_inner_x[-1]):.3f}")

        print()


        # Visualize the results
        plot_iterations("Quadratic Program", history_inner_obj, history_outer_obj)
        plot_feasible_regions_3d(history_inner_x)

    def test_linear_program(self):

        # Set up the problem constraints
        inequality_constraints = [lp_ineq_1, lp_ineq_2, lp_ineq_3, lp_ineq_4]
        equality_matrix = np.array([])
        equality_rhs = np.array([])
        initial_guess = np.array([0.5, 0.75], dtype=np.float64)



        # Plot the new graph

        # Execute the solver
        history_inner_x, history_inner_obj, _, history_outer_obj,_ = interior_pt(lp_func, inequality_constraints,
                                                                               equality_matrix, equality_rhs, initial_guess)

        # Display the results
        print(f"Final solution: {history_inner_x[-1]}")
        print(f"Objective value at the final solution: {lp_func(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 1 value at the final solution: {lp_ineq_1(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 2 value at the final solution: {lp_ineq_2(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 3 value at the final solution: {lp_ineq_3(history_inner_x[-1])[0]:.3f}")
        print(f"Constraint 4 value at the final solution: {lp_ineq_4(history_inner_x[-1])[0]:.3f}")
        print()

        # Visualize the results
        plot_iterations("Linear Program", history_inner_obj, history_outer_obj)
        plot_feasible_set_2d(history_inner_x)


if __name__ == '__main__':
    unittest.main()