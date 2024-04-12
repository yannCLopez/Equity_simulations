# Program: To find the G and b that result in the biggest difference in slopes for central and periphery nodes.
# References: Building on Anant's modification of Yann's code for a core-periphery network.
# Date: 4-12-2024 (American date format).

#To do: Add the contraint that the diagonal of G should be set to 0.

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, minimize
import matplotlib.pyplot as plt

# Parameters
num_agents = 12

G = np.array(
    [
        [0, 1, 3, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0],
        [3, 3, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01],
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# Add tiny random positive noise to all nonzero entries to prevent invertibility problem
for i in range(G.shape[0]):  # Iterate over rows
    for j in range(G.shape[1]):  # Iterate over columns
        if G[i, j] != 0:  # Check if the element is not zero
            G[i, j] += np.random.uniform(0, 0.0002)  # Add random noise

p = 0.2
a = 0.6  # we set all entries of b_base_central equal to a. We also scale the changes in b_base_periphery by a.

rand_bound = 0.1  # The distance from an individual productivity of 1. Essentially b is sampled from [1-rand_bound,1+rand_bound]


b_base = np.ones(num_agents)
# b_base[0] = a
# b_base[1] = 1
b_base[2] = 0.8
b_base[6] = 1.08

epsilon = 0.02  # Small perturbation magnitude


def compute_a_star(t, p, G, b):
    n = len(t)
    I = np.eye(n)
    T_sqrt = np.diag(np.sqrt(t))
    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ b
    return a_star


################################################################
# Compute results original and perturbed for central and periphery nodes, separately.
# Function to calculate derivatives A and B


##PERIPHERY NODES
def calculate_derivatives(G, b_base, num_agents, epsilon):
    deltab = np.full(num_agents, np.random.uniform(0, epsilon))  # Uniform small shock applied to all agents
    #deltab = np.full(num_agents, 0.02)  # Uniform small shock applied to all agents

    
    b_perturbed = b_base + deltab

    def objective_function(t, p, G, b):
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

    # Get original values. Bounds for t are (0, 1)
    result_original = differential_evolution(
        lambda t: objective_function(t, p=p, G=G, b=b_base),
        [(0, 1)] * num_agents,
        constraints=[nonlinear_constraint],
    )
    optimal_t_original_all = result_original.x
    a_star_optimal_original = compute_a_star(
        optimal_t_original_all, p, G, b_base
    )
    kappa_all_agents_original = (
        b_base + G @ a_star_optimal_original
    )
    kappa_original_periphery = kappa_all_agents_original[6]
    optimal_t_original_periphery = optimal_t_original_all[6]

    kappa_original_central = kappa_all_agents_original[0]
    optimal_t_original_central = optimal_t_original_all[0]



    # Get perturbed values 
    result_perturbed = differential_evolution(
        lambda t: objective_function(t, p=p, G=G, b=b_perturbed),
        [(0, 1)] * num_agents,
        constraints=[nonlinear_constraint],
    )
    optimal_t_perturbed_all = result_perturbed.x

    a_star_optimal_perturbed_periphery = compute_a_star(
        optimal_t_perturbed_all, p, G, b_perturbed
    )
    kappa_all_agents_perturbed = (
        b_perturbed + G @ a_star_optimal_perturbed_periphery
    )
    kappa_perturbed_periphery = kappa_all_agents_perturbed[6]
    optimal_t_perturbed_periphery = optimal_t_perturbed_all[6]

    kappa_perturbed_central = kappa_all_agents_perturbed[0]
    optimal_t_perturbed_central = optimal_t_perturbed_all[0]

    # Calculate derivatives. Bounds for t are (0, 1)
    A = (
        (optimal_t_perturbed_central - optimal_t_original_central)
        / (kappa_perturbed_central - kappa_original_central)
        if kappa_perturbed_central != kappa_original_central
        else float("inf")
    )
    B = (
        (optimal_t_perturbed_periphery - optimal_t_original_periphery)
        / (kappa_perturbed_periphery - kappa_original_periphery)
        if kappa_perturbed_periphery != kappa_original_periphery
        else float("inf")
    )
    return A, B


# Example call to the function
A, B = calculate_derivatives(G, b_base, num_agents, epsilon)
print("Derivative A (central):", A)
print("Derivative B (periphery):", B)


# OPTIMISATION MACHINE


def optimization_objective(variables, num_agents, G_base, b_base):
    x, y, z = variables
    # Apply x to central nodes of G
    G_base[0, 1] += x
    G_base[0, 2] += x

    G_base[1, 0] += x
    G_base[2, 0] += x

    G_base[1, 0] += x
    G_base[1, 2] += x

    G_base[0, 1] += x
    G_base[2, 1] += x

    G_base[2, 0] += x
    G_base[2, 1] += x

    G_base[1, 2] += x
    G_base[0, 2] += x

    # Apply y to periphery nodes of G
    for i in range(3, num_agents):
        G_base[i, :] += y
        G_base[:, i] += y

    # Apply z to b
    b_new = b_base + z

    # Ensure G remains symmetric and non-negative
    G_new = np.maximum(G_base, 0)
    G_new = (G_new + G_new.T) / 2

    A, B = calculate_derivatives(G_new, b_new, num_agents, epsilon)
    return -(A - B)  # Objective is to maximize the absolute difference


# Adjust initial guess and bounds for x, y, and z
initial_guess = [0, 0, 0]  # Starting from no change
bounds = [
    (-0.01, 0.01),
    (-0.01, 0.01),
    (-0.01, 0.01),
]  # Small bounds for changes in x, y, and z

# Perform the optimization with the new setup
result = minimize(
    optimization_objective,
    initial_guess,
    args=(num_agents, G, b_base),
    bounds=bounds,
    method="L-BFGS-B",
)

# Extract optimised values
optimised_x, optimised_y, optimised_z = result.x

# Calculate the optimised G and b based on the optimised values of x, y, and z
G_optimised = np.copy(
    G
)  # Start with a copy of the original G to apply the optimised changes

# Apply optimised x to central nodes of G
G_optimised[0, 1] += optimised_x
G_optimised[0, 2] += optimised_x

G_optimised[1, 0] += optimised_x
G_optimised[2, 0] += optimised_x

G_optimised[1, 0] += optimised_x
G_optimised[1, 2] += optimised_x

G_optimised[0, 1] += optimised_x
G_optimised[2, 1] += optimised_x

G_optimised[2, 0] += optimised_x
G_optimised[2, 1] += optimised_x

G_optimised[1, 2] += optimised_x
G_optimised[0, 2] += optimised_x
# Apply optimised y to periphery nodes of G
for i in range(3, num_agents):
    G_optimised[i, :] += optimised_y
    G_optimised[:, i] += optimised_y

# Ensure G remains symmetric and non-negative
G_optimised = np.maximum(G_optimised, 0)
G_optimised = (G_optimised + G_optimised.T) / 2

# Apply optimised z to b
b_optimised = b_base + optimised_z

print("optimised G:", G_optimised)
print("optimised b:", b_optimised)
