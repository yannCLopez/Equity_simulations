# Import necessary libraries
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.optimize import differential_evolution

# Initialize parameters
num_agents = 4
p = 0.05
gamma = 0.95
perturbation = 0.02  # Perturbation amount for productivity

# Initialize arrays to store results
total_t_matrix = np.zeros((num_agents, num_agents))
total_a_star_matrix = np.zeros((num_agents, num_agents))
total_kappa_matrix = np.zeros((num_agents, num_agents))

# Initialize the graph matrix G
G = np.ones((num_agents, num_agents))
np.fill_diagonal(G, 0)  # Remove self-loops
G = 0.1 * G  # Scale the graph matrix

# Function to compute a_star using optimal t values
def compute_a_star(t, p, G, k):
    n = len(t)
    I = np.eye(n)
    T_diag = np.diag(t ** gamma)
    a_star = p * np.linalg.inv(I - p * T_diag @ G) @ T_diag @ k
    return a_star

# Define the objective function for optimization
def objective_function(t, p, G, k):
    n = len(t)
    I = np.eye(n)
    T_diag = np.diag(t ** gamma)
    a_star = p * np.linalg.inv(I - p * T_diag @ G) @ T_diag @ k
    Y_a = k @ a_star + 0.5 * a_star.T @ G @ a_star
    Y = min(Y_a, 1)
    return (-1 + np.sum(t)) * p * Y

# Set bounds for each t_i in the range [0, 1]
bounds = [(0, 1) for _ in range(num_agents)]

# Define a constraint to ensure that the sum of all t_i values does not exceed 1
constraint = {'type': 'ineq', 'fun': lambda t: 1 - np.sum(t)}

k_base = np.ones(num_agents)
def constraint_sum(t):
    return 1 - np.sum(t)
nonlinear_constraint = NonlinearConstraint(constraint_sum, -np.inf, 1)

# Base optimization to find initial t values without perturbation
result_base = differential_evolution(objective_function, bounds, args=(p, G, k_base), constraints=[nonlinear_constraint],popsize=200)

t_base = result_base.x
a_star_base = compute_a_star(t_base, p, G, k_base)
kappa_base = k_base + G @ a_star_base

# Initial guess for optimization
initial_guess = t_base


num_simulations=100;
for simulation in range(num_simulations):

    # Perturb productivity and compute derivatives
    for i in range(num_agents):
        k = np.copy(k_base)
        k[i] += perturbation
        #result = differential_evolution(objective_function, bounds, args=(p, G, k), constraints=[nonlinear_constraint], popsize=100)

        result = minimize(objective_function, initial_guess, args=(p, G, k),method='BFGS')
        optimal_t = result.x
        a_star_optimal = compute_a_star(optimal_t, p, G, k)

        # Store results and compute derivatives
        optimal_t_matrix = (optimal_t - t_base) / perturbation
        optimal_a_star_matrix = (a_star_optimal - a_star_base) / perturbation
        kappa_matrix = (k + G @ a_star_optimal - kappa_base) / perturbation

        # Accumulate results
        total_t_matrix[i, :] += optimal_t_matrix
        total_a_star_matrix[i, :] += optimal_a_star_matrix
        total_kappa_matrix[i, :] += kappa_matrix

average_t_matrix = total_t_matrix / num_simulations
average_a_star_matrix = total_a_star_matrix / num_simulations
average_kappa_matrix = total_kappa_matrix / num_simulations

# Print the results as approximate derivatives
print("Base t:")
print(t_base)
print("\nApproximate Derivatives of Optimal t Matrix:")
print(average_t_matrix)
print("\nApproximate Derivatives of Optimal a_star Matrix:")
print(average_a_star_matrix)
print("\nApproximate Derivatives of Kappa Matrix:")
print(average_kappa_matrix)