import matplotlib.pyplot as plt
import unittest
import numpy as np
from matplotlib import pyplot as plt
import warnings

from matplotlib.colors import LogNorm

from examples import example_1, example_2, example_3, rosenbrock_function, linear_function, exponential_function
from src.unconstrained_min import LineSearchOptimizer
from matplotlib import MatplotlibDeprecationWarning


class TestOptimizationAlgorithms(unittest.TestCase):
    def setUp(self):
        # Tolerances and iterations setup
        self.obj_tol = 1e-08
        self.param_tol = 1e-12
        self.max_iter_standard = 100
        self.max_iter_rosenbrock = 10000
        self.initial_point_standard = np.array([1, 1])
        self.initial_point_rosenbrock = np.array([-1, 2])
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    def test_functions(self):
        functions = [
            (example_1, self.initial_point_standard, self.max_iter_standard, "Example 1"),
            (example_2, self.initial_point_standard, self.max_iter_standard, "Example 2"),
            (example_3, self.initial_point_standard, self.max_iter_standard, "Example 3"),
            (rosenbrock_function, self.initial_point_rosenbrock, self.max_iter_rosenbrock, "Rosenbrock Function"),
            (linear_function, self.initial_point_standard, self.max_iter_standard, "Linear Function"),
            (exponential_function, self.initial_point_standard, self.max_iter_standard, "Exponential Function")
        ]
        for func, initial_point, max_iter, example_name in functions:
            with self.subTest(func=func.__name__):
                optimizer = LineSearchOptimizer(func, grad=lambda x: func(x, need_hessian=False)[1],
                                                hess=lambda x: func(x, need_hessian=True)[2])
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                results = {}
                for method in ['gradient_descent', 'newton']:
                    final_x, final_f, success, iterations = optimizer.optimize(initial_point, method=method,
                                                                               obj_tol=self.obj_tol,
                                                                               param_tol=self.param_tol,
                                                                               max_iter=max_iter)
                    results[method] = {
                        'final_x': final_x,
                        'final_f': final_f,
                        'success': success,
                        'iterations': iterations
                    }
                    print(f"{example_name} - Method: {method}, Final x: {final_x}, Final f: {final_f}, Success: {success}, Iterations: {len(iterations)}")

                    # Plot path on the contour plot
                    x_vals, y_vals = zip(*[it[1] for it in iterations])
                    axes[0].plot(x_vals, y_vals, label=f"{method} Path")
                    # Plot function values
                    axes[0].set_xlim(-3, 3)  # Set x-axis limits
                    axes[0].set_ylim(-3, 3)  # Set y-axis limits
                    f_vals = [it[2] for it in iterations]
                    axes[1].plot(f_vals, label=f"{method} Function Value")

                # Contour plot setup
                x_range = np.linspace(-3, 3, 400)
                y_range = np.linspace(-3, 3, 400)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.vectorize(lambda x, y: func(np.array([x, y]), need_hessian=False)[0])(X, Y)

                if example_name == "Rosenbrock Function":
                    axes[0].set_xlim(-3, 3)  # Set x-axis limits
                    axes[0].set_ylim(-3,3)  # Set y-axis limits
                    x_range = np.linspace(-3, 3, 1000)
                    y_range = np.linspace(-3, 3, 1000)
                    X, Y = np.meshgrid(x_range, y_range)
                    Z = np.vectorize(lambda x, y: func(np.array([x, y]), need_hessian=False)[0])(X, Y)
                    axes[0].contour(X, Y, Z, levels=300)
                    axes[0].set_title(f'Contour Plot for {func.__name__}')
                    axes[0].set_xlabel('x1')
                    axes[0].set_ylabel('x2')
                    axes[0].legend()
                if example_name == 'Exponential Function':
                    axes[0].set_xlim(-1, 1)  # Set x-axis limits
                    axes[0].set_ylim(-1, 1)  # Set y-axis limits
                    x_range = np.linspace(-1, 1, 400)
                    y_range = np.linspace(-1, 1, 400)
                    X, Y = np.meshgrid(x_range, y_range)
                    Z = np.vectorize(lambda x, y: func(np.array([x, y]), need_hessian=False)[0])(X, Y)
                    axes[0].contour(X, Y, Z, levels=200, extend='both')
                    axes[0].set_title(f'Contour Plot for {func.__name__}')
                    axes[0].set_xlabel('x1')
                    axes[0].set_ylabel('x2')
                    axes[0].legend()



                else:
                    axes[0].contour(X, Y, Z, levels=100, extend='both')
                    axes[0].set_title(f'Contour Plot for {func.__name__}')
                    axes[0].set_xlabel('x1')
                    axes[0].set_ylabel('x2')
                    axes[0].legend()




                # Function values plot setup
                axes[1].set_title('Function Values vs Iteration')
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Function Value')
                axes[1].legend()

                plt.tight_layout()
                plt.show()


if __name__ == '__main__':
    unittest.main()