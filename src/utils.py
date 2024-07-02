import numpy as np
import matplotlib.pyplot as plt
def plot_duality_gap_vs_iterations(results):
    plt.figure(figsize=(10, 6))
    for mu, duality_gaps, total_iterations in results:
        cumulative_iterations = list(range(len(duality_gaps)))
        plt.semilogy(cumulative_iterations, duality_gaps, label=f'μ = {mu}')

    plt.xlabel('Total Number of iterations')
    plt.ylabel('Duality gap')
    plt.legend()
    plt.title('Duality Gap vs. Total Number of Newton Iterations')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
def plot_iterations(
        program_name,
        inner_values,
        outer_values
):
    figure, axes = plt.subplots()
    axes.plot(range(len(inner_values)), inner_values, color='crimson', label="Inner Iterations")
    axes.plot(range(len(outer_values)), outer_values, color='navy', label="Outer Iterations")

    axes.legend()
    axes.set_title(f"{program_name} Program Optimization Progress")
    axes.set_xlabel("Iteration Count")
    axes.set_ylabel("Objective Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_feasible_regions_3d(
        trajectory
):
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    trajectory_array = np.array(trajectory)

    axes.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color="palegreen", alpha=0.6)
    axes.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], color="purple", marker="o", linestyle="-.", linewidth=2, markersize=5, label="Optimization Path")
    axes.scatter(trajectory_array[-1][0], trajectory_array[-1][1], trajectory_array[-1][2], s=100, c="red", edgecolors="black", label="Optimal Point")

    axes.set_title("Quadratic Program: Feasible Region and Optimization Path", fontsize=14)
    axes.set_xlabel("X-axis", fontsize=12)
    axes.set_ylabel("Y-axis", fontsize=12)
    axes.set_zlabel("Z-axis", fontsize=12)
    plt.legend(fontsize=10)
    axes.view_init(30, 60)
    plt.show()

def plot_feasible_set_2d(
        trajectory

):
    x_range = np.linspace(-2, 4, 500)
    x_grid, y_grid = np.meshgrid(x_range, x_range)
    plt.figure(figsize=(12, 9))
    plt.imshow(
        ((y_grid >= -x_grid + 1) & (y_grid <= 1) & (x_grid <= 2) & (y_grid >= 0)).astype(int),
        extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
        origin="lower",
        cmap="YlGnBu",
        alpha=0.4,
    )

    x_line = np.linspace(0, 4, 1000)
    y_constraint1 = -x_line + 1
    y_constraint2 = np.ones(x_line.size)
    y_constraint3 = np.zeros(x_line.size)

    x_trajectory = [point[0] for point in trajectory]
    y_trajectory = [point[1] for point in trajectory]
    plt.plot(x_trajectory, y_trajectory, label=" Path", color="magenta", marker="*", linestyle=":", linewidth=2, markersize=8)
    plt.scatter(x_trajectory[-1], y_trajectory[-1], label="Minimizer", color="red", s=150, edgecolors="black", zorder=5)

    plt.plot(x_line, y_constraint1, color='darkgreen', linewidth=2, label="y = -x + 1")
    plt.plot(x_line, y_constraint2, color='darkblue', linewidth=2, label="y = 1")
    plt.plot(x_line, y_constraint3, color='darkred', linewidth=2, label="y = 0")
    plt.plot(np.ones(x_line.size) * 2, x_line, color='darkorange', linewidth=2, label="x = 2")

    plt.xlim(0, 3.5)
    plt.ylim(0, 2.5)
    plt.legend(fontsize=10)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.title("Linear Program: Feasible Region and Optimization Path", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()



