import numpy as np
import networkx as nx
from scipy.optimize import differential_evolution, LinearConstraint
from scipy.stats import spearmanr
import pandas as pd

def generate_G(n):
    connected = False
    while not connected:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.choice([True, False]):
                    weight = np.random.uniform(0.2, 1)
                    G.add_edge(i, j, weight=weight)
        connected = nx.is_connected(G)
    A = nx.to_numpy_array(G)
    return A

def objective_function(t, p, G):
    n = len(t)
    I = np.eye(n)
    T_sqrt = np.diag(np.sqrt(np.maximum(t, np.zeros(n))))
    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ np.ones(n)
    Y_a = np.ones(n) @ a_star + 0.5 * a_star.T @ G @ a_star
    P_uncensored = p * Y_a
    P_real = min(P_uncensored, 1)
    return -(1 - np.sum(t)) * P_real

def compute_a_star(t, p, G):
    n = len(t)
    I = np.eye(n)
    T_sqrt = np.diag(np.sqrt(np.maximum(t, np.zeros(n))))
    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ np.ones(n)
    return a_star

def matrix_to_string(matrix):
    return ';'.join(','.join('%0.4f' %x for x in row) for row in matrix)

# Setup for the simulation
n = 7
p = 0.3
iterations = 1000000  # Adjusted to a large number for extensive simulation
batch_size = 100  # Define a batch size for incremental saving

# File path for saving
file_path = "/Users/yuchi/Library/CloudStorage/GoogleDrive-hsiehyuchi.tw@gmail.com/My Drive/0_Work/_Ben/Equity/optimization_results.csv"  # Adjust the path accordingly

for i in range(iterations):
    G = generate_G(n)
    bounds = [(0, 1) for _ in range(n)]
    constraint = LinearConstraint(np.ones((1, n)), -np.inf, np.array([1]))
    result = differential_evolution(objective_function, bounds, args=(p, G), constraints=[constraint], maxiter=1000, popsize=15)
    
    optimal_t = result.x
    a_star_optimal = compute_a_star(optimal_t, p, G)
    spearman_corr, _ = spearmanr(optimal_t, a_star_optimal)
    
    # Prepare data for a single simulation
    data = {
        "G": matrix_to_string(G),
        "Normalized t": matrix_to_string([optimal_t / np.sum(optimal_t)]),
        "Normalized a_star": matrix_to_string([a_star_optimal / np.sum(a_star_optimal)]),
        "Std Dev of Normalized t": np.std(optimal_t / np.sum(optimal_t)),
        "Std Dev of Normalized a_star": np.std(a_star_optimal / np.sum(a_star_optimal)),
        "Sum of optimal_t": np.sum(optimal_t),
        "Optimal P(Y)": min(p * (np.ones(n) @ a_star_optimal + 0.5 * a_star_optimal.T @ G @ a_star_optimal), 1),
        "Principal profit objective value": -result.fun,
        "Spearman Rank Correlation": spearman_corr
    }
    df = pd.DataFrame([data])
    
    # Append results to the CSV file
    if i % batch_size == 0 and i != 0:  # Check if it's time to save based on batch_size and not the first iteration
        df.to_csv(file_path, mode='a', header=False, index=False)
        print(i)
    elif i == 0:  # For the first iteration, write with header
        df.to_csv(file_path, mode='w', header=True, index=False)
    else:  # For other iterations within a batch, append without writing the header
        df.to_csv(file_path, mode='a', header=False, index=False)

print("Completed simulations and incremental saving to 'optimization_results.csv'.")