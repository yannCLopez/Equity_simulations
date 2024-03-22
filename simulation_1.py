import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def generate_G(n, seed):
    """
    Generates a symmetric, connected, and unweighted graph G for n agents.

    Parameters:
    - n (int): Number of agents (nodes in the graph).
    - seed (int): Random seed for reproducibility.

    Returns:
    - numpy.ndarray: Adjacency matrix of the generated graph G.
    """
    # np.random.seed(seed)  # Fix the random seed for reproducibility

    # Ensure the generated graph is connected
    while True:
        # Generate a random graph
        G = nx.erdos_renyi_graph(n, 0.5, seed=np.random.randint(0, 10000))

        # Check if the graph is connected
        if nx.is_connected(G):
            break

    # Convert to adjacency matrix and then to numpy array
    A = nx.adjacency_matrix(G).toarray()

    # Ensure the graph is unweighted
    A[A > 0] = 1

    return A

def compute_a_star(sigma, G, p):
    """
    Computes the optimal actions a* for agents given a contract sigma, a connectivity graph G, and a parameter p.

    Parameters:
    - sigma (numpy.ndarray): A vector of length n representing the contract, where sigma[i] is the transfer agent i receives.
    - G (numpy.ndarray): The adjacency matrix of the unweighted, symmetric, and connected graph representing agent connectivity.
    - p (float): A parameter in the range [0, 1].

    Returns:
    - numpy.ndarray: The optimal action vector a* for the agents.
    """
    n = len(sigma)  # Number of agents
    t_sqrt = np.sqrt(sigma)  # Transform sigma into t by taking the square root of each element

    # Create the diagonal matrix diag(t)
    diag_t = np.diag(t_sqrt)

    # Calculate the term (I - p * diag(t) * G) and its inverse
    I = np.identity(n)  # Identity matrix of size n
    inverse_term = np.linalg.inv(I - p * np.dot(diag_t, G))

    # Calculate the optimal action vector a*
    a_star = p * np.dot(np.dot(inverse_term, diag_t), np.ones(n))

    return a_star

def compute_kappa_functions(sigma, G, p, a_star):
    """
    Computes kappa_i * Bar{kappa}_i * u_i'(sigma_i*(s)) for all agents i, given sigma, G, p, and a_star.

    Parameters:
    - sigma (numpy.ndarray): Contract vector for all agents.
    - G (numpy.ndarray): Adjacency matrix of the unweighted, symmetric, and connected graph.
    - p (float): Parameter in the range [0, 1].
    - a_star (numpy.ndarray): Optimal action vector for all agents.

    Returns:
    - numpy.ndarray: Computed values for kappa_i * Bar{kappa}_i * u_i'(sigma_i*(s)) for each agent i.
    """
    n = len(sigma)
    t_sqrt = np.sqrt(sigma)

    # Gradient of Y at a*
    grad_Y = 1 + np.dot(G, a_star)

    # Computing kappa
    H_inv_sqrt = np.identity(n)  # H is the identity matrix, so its inverse square root is also an identity matrix
    kappa = np.dot(H_inv_sqrt, grad_Y)

    # Computing U
    Y_a_star = np.dot(np.ones(n), a_star) + 0.5 * np.dot(a_star.T, np.dot(G, a_star))
    U_diagonal = p * (Y_a_star * t_sqrt - np.power(a_star, 2) / 2)
    U = np.diag(U_diagonal)

    # Computing Bar{kappa}
    bar_kappa_T = np.dot(kappa.T, np.linalg.inv(np.identity(n) - np.dot(U, G)))

    # Computing u_i'(sigma_i*(s))
    ui_prime = p * (np.dot(np.ones(n), a_star) + 0.5 * np.dot(a_star.T, np.dot(G, a_star))) * (0.5 / t_sqrt)

    # Final computation for each agent
    result = kappa * bar_kappa_T * ui_prime

    return result

seed=4
n = 5  # Number of agents
G = generate_G(n, seed)

p = 0.7  # Given parameter p in the range [0, 1]
sigma = np.array([1, 2, 3, 4, 5])  # Example contract vector sigma

a_star = compute_a_star(sigma, G, p)
result = compute_kappa_functions(sigma, G, p, a_star)
objective = sum((result[i] - result[j])**2 for i in range(len(result)) for j in range(i + 1, len(result)))
print(objective)


# Objective function
def objective_function(sigma, G, p):
    a_star = compute_a_star(sigma, G, p)
    result = compute_kappa_functions(sigma, G, p, a_star)
    objective = sum((result[i] - result[j])**2 for i in range(len(result)) for j in range(i + 1, len(result)))
    return objective

# Constraint: sum(sigma) = 1
constraints = ({'type': 'eq', 'fun': lambda sigma: np.sum(sigma) - 0.7})

# Bounds for sigma_i: 0 < sigma_i < 1
bounds = [(0.01, 1) for _ in range(n)]

# Initial sigma
sigma_initial = [.05, .05, .5, .3, .4]

# Optimization
opt_result = minimize(
    objective_function, sigma_initial, args=(G, p),
    method='SLSQP', bounds=bounds,
    # constraints=constraints,
    options={'maxiter': 100000000, 'ftol': 1e-8}
)

if opt_result.success:
    optimized_sigma = opt_result.x
    print("Optimized sigma:", optimized_sigma)
else:
    print("Optimization failed:", opt_result.message)

 
a_star = compute_a_star(optimized_sigma, G, p)
result = compute_kappa_functions(optimized_sigma, G, p, a_star)
objective = sum((result[i] - result[j])**2 for i in range(len(result)) for j in range(i + 1, len(result)))
print("Objective:", objective)

# Assuming a_star is your optimized action vector from previous computations
# And G is the graph adjacency matrix

# Calculate 1 + Ga*
transformed_actions = 1 + np.dot(G, a_star)

# Compute Spearman rank correlation
spearman_corr, _ = spearmanr(transformed_actions, a_star)

print("Spearman Rank Correlation (1 + Ga*, a*):", spearman_corr)

def visualize_graph(G):
    """
    Visualizes the graph G using NetworkX and Matplotlib.

    Parameters:
    - G (numpy.ndarray): The adjacency matrix of the graph.
    """
    # Convert the numpy adjacency matrix to a NetworkX graph
    # Use from_numpy_array if from_numpy_matrix is not available
    G_nx = nx.from_numpy_array(G)

    # Draw the graph
    plt.figure(figsize=(4, 4))
    nx.draw(G_nx, with_labels=True, node_color='skyblue', node_size=700,
            edge_color='k', linewidths=1, font_size=15,
            pos=nx.spring_layout(G_nx, seed=42))
    plt.title('Graph Visualization')
    plt.show()




def run_simulation(n, p, bounds, max_iterations, ftol, constraints):
    success_count = 0
    spearman_correlations = []
    detailed_results = []  # To store detailed results for each simulation

    for _ in range(max_iterations):
        G = generate_G(n, np.random.randint(0, 10000))  # Generate G with a random seed
        sigma_initial = np.full(n, .5/n)  # Adjusted initial sigma to ensure sum(sigma) = .5

        # Optimization
        opt_result = minimize(
            objective_function, sigma_initial, args=(G, p),
            method='SLSQP', bounds=bounds,
            constraints=constraints,
            options={'maxiter': 10000, 'ftol': ftol}
        )

        if opt_result.success:
            success_count += 1
            optimized_sigma = opt_result.x
            a_star = compute_a_star(optimized_sigma, G, p)
            transformed_actions = 1 + np.dot(G, a_star)

            # Compute Spearman rank correlation
            spearman_corr, _ = spearmanr(transformed_actions, a_star)
            spearman_correlations.append(spearman_corr)

            # Store detailed results
            detailed_results.append((G, spearman_corr, optimized_sigma, a_star, transformed_actions))

    # Find the G with the least Spearman Rank Correlation
    if detailed_results:
        sorted_results = sorted(detailed_results, key=lambda x: x[1])

        # Calculate the index for the 20th percentile
        n = len(sorted_results)
        index = int(0.2 * n)

        # Handle cases where the calculated index is equal to the length of the list
        index = min(index, n - 1)

        # Select the item at the 5th percentile
        percentile_20th_result = sorted_results[index]
        G_least_corr, least_corr, sigma_least_corr, a_star_least_corr, transformed_actions_least_corr = percentile_20th_result

        # Print details for the G with the least Spearman Rank Correlation
        print("G with the least Spearman Rank Correlation:")
        print("Spearman Rank Correlation:", least_corr)
        print("Optimized sigma:", sigma_least_corr)
        print("a_star:", a_star_least_corr)
        print("1 + Ga*:", transformed_actions_least_corr)
        print("G:\n", G_least_corr)

        # Visualize the graph G with the least Spearman Rank Correlation
        visualize_graph(G_least_corr)

    return success_count, spearman_correlations, detailed_results

# Run simulations
success_count, spearman_correlations, detailed_results = run_simulation(n, p, bounds, 100, 1e-8, constraints)

# Visualization
plt.hist(spearman_correlations, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Spearman Rank Correlations')
plt.xlabel('Spearman Rank Correlation')
plt.ylabel('Frequency')
plt.show()
