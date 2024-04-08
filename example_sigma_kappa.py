import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt

num_agents = 12
p = 0.2
# G = np.array([[0, 1, 1, 0, 1],[1, 0, 0, 1, 1],[1, 0, 0, 1, 0],[0, 1, 1, 0, 1],[1, 1, 0, 1, 0]])
# G=np.array([[0.1, 0.02, 0.03], [0.02, 0.2, 0.03], [0.03, 0.03, 0.3]])


G = [
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


num_iter = 1
central_equity_ratio = np.zeros(num_iter)
peripheral_equity_ratio_to_central = np.zeros(num_iter)
peripheral_equity_ratio_to_peripheral = np.zeros(num_iter)

for iter in range(num_iter):
    b = 1 + np.random.uniform(-0.2, 0.2, num_agents)

    # Adapted objective function for your problem
    def objective_function(t, p=p, G=G, b=b):
        n = len(t)
        I = np.eye(n)
        T_sqrt = np.diag(np.sqrt(t))
        a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ b
        Y_a = b @ a_star + 0.5 * a_star.T @ G @ a_star
        Y = min(Y_a, 1)
        return (-1 + np.sum(t)) * p * Y

    # Bounds for t
    n = len(G)  # Dimensionality of t, adjust based on G if needed
    bounds = [(0, 1) for _ in range(n)]

    # Constraint ensuring sum(t_i) <= 1
    def constraint_sum(t):
        return 1 - np.sum(t)

    nonlinear_constraint = NonlinearConstraint(constraint_sum, -np.inf, 1)

    # Performing differential evolution with the adapted objective function and constraints
    result = differential_evolution(
        objective_function, bounds, constraints=[nonlinear_constraint]
    )

    def compute_a_star(t, p, G, b):
        n = len(t)
        I = np.eye(n)
        T_sqrt = np.diag(np.sqrt(t))
        a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ b
        return a_star

    optimal_t = (
        result.x
    )  # This assumes 'result' is the output of your differential_evolution call
    a_star_optimal = compute_a_star(optimal_t, p, G, b)

# Compute the derivative of Y_a with respect to each entry of a_star_optimal
kappa = b + G @ a_star_optimal

plt.figure(figsize=(10, 6))
plt.plot(kappa[2:], optimal_t[2:], 'o')
plt.plot(kappa[:3], optimal_t[:3], 'o', color='#32CD32')
plt.title('Figure 1')
# Update the labels with LaTeX formatting
plt.xlabel(r'$\kappa$', fontsize=14)  # LaTeX formatted x-axis label
plt.ylabel(r'$\sigma^*$', fontsize=14)  # LaTeX formatted y-axis label
plt.grid(True)
plt.show()


# Compute the line of best fit for the blue dots (excluding the first three points).
m_blue, b_blue = np.polyfit(kappa[3:], optimal_t[3:], 1)

# Compute the line of best fit for the first three points (green dots).
m_green, b_green = np.polyfit(kappa[:3], optimal_t[:3], 1)

# Create a figure for plotting.
plt.figure(figsize=(10, 6))

# Plot the blue dots and the line of best fit for them.
plt.plot(kappa[3:], optimal_t[3:], 'o', color='blue', label='Data (blue)')
plt.plot(kappa[3:], m_blue * kappa[3:] + b_blue, '-', color='blue', label='Best Fit (blue)')

# Plot the first three points (green dots) and the line of best fit for them.
plt.plot(kappa[:3], optimal_t[:3], 'o', color='#32CD32', label='Data (green)')
plt.plot(kappa[:3], m_green * kappa[:3] + b_green, '-', color='#32CD32', label='Best Fit (green)')

plt.title('Figure 2')
# Update the labels with LaTeX formatting
plt.xlabel(r'$\kappa$', fontsize=14)  # LaTeX formatted x-axis label
plt.ylabel(r'$\sigma^*$', fontsize=14)  # LaTeX formatted y-axis label
plt.grid(True)
plt.show()
