#Program:  Perturbs the entries of G upward by perturbation,and creates a "tensor" (three-dimensional matrix object) to keep track of the effect of each perturbation
#in some entry (k, ell), on each person i.
#References: Building on Ben's code for perturbations in k
Date: 04-19-2024 


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
G_base = np.ones((num_agents, num_agents))
np.fill_diagonal(G_base, 0)  # Remove self-loops
G_base = 0.1 * G_base  # Scale the graph matrix

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
result_base = differential_evolution(objective_function, bounds, args=(p, G_base, k_base), constraints=[nonlinear_constraint],popsize=200)

t_base = result_base.x
print ("t base", t_base)
a_star_base = compute_a_star(t_base, p, G_base, k_base)
kappa_base = k_base + G_base @ a_star_base

# Initial guess for optimization
initial_guess = t_base


num_simulations=4

# Initialize tensors to store the effects of perturbations
# Dimensions: num_agents x num_agents x num_agents
# Each entry (i, j, k) in these tensors represents the effect of perturbing G[j, k] on agent i
effects_t_tensor = np.zeros((num_agents, num_agents, num_agents))
effects_a_star_tensor = np.zeros((num_agents, num_agents, num_agents))
effects_kappa_tensor = np.zeros((num_agents, num_agents, num_agents))

for simulation in range(num_simulations):

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:  # Skip diagonal entries to avoid self-perturbation
                G_perturbed = np.copy(G_base)
                G_perturbed[i, j] += perturbation  # Perturb the entry (i, j) of G

                # Perform optimization with the perturbed G
                result = minimize(objective_function, initial_guess, args=(p, G_perturbed, k_base), method='BFGS')
                optimal_t_perturbed = result.x
                print(f"optimal t perturbed for [{i}], [{j}]", optimal_t_perturbed)
                a_star_perturbed = compute_a_star(optimal_t_perturbed, p, G_perturbed, k_base)
                kappa_perturbed = k_base + G_perturbed @ a_star_perturbed

                # Calculate the effects of the perturbation
                effects_t = (optimal_t_perturbed - t_base) / perturbation
                effects_a_star = (a_star_perturbed - a_star_base) / perturbation
                effects_kappa = (kappa_perturbed - kappa_base) / perturbation

                # Store the effects in the tensors
                effects_t_tensor[:, i, j] = effects_t
                effects_a_star_tensor[:, i, j] = effects_a_star
                effects_kappa_tensor[:, i, j] = effects_kappa
           
 # Calculate the average effects of perturbations in G on t, a_star, and kappa

average_effects_t_tensor = np.mean(effects_t_tensor, axis=(1, 2))
average_effects_a_star_tensor = np.mean(effects_a_star_tensor, axis=(1, 2))
average_effects_kappa_tensor = np.mean(effects_kappa_tensor, axis=(1, 2))

print("\nAverage Effects on t:")
print(average_effects_t_tensor)
print("\nAverage Effects on a_star:")
print(average_effects_a_star_tensor)
print("\nAverage Effects on kappa:")
print(average_effects_kappa_tensor, "\n")


