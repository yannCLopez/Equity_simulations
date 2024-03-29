import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

p=0.7
G = np.array([[0, 1, 1, 0, 1],[1, 0, 0, 1, 1],[1, 0, 0, 1, 0],[0, 1, 1, 0, 1],[1, 1, 0, 1, 0]])
#G=np.array([[0.1, 0.02, 0.03], [0.02, 0.2, 0.03], [0.03, 0.03, 0.3]])

# Adapted objective function for your problem
def objective_function(t, p=p, G=G):
    n = len(t)
    I = np.eye(n)
    T_sqrt = np.diag(np.sqrt(t))
    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ np.ones(n)
    Y_a = np.ones(n) @ a_star + 0.5 * a_star.T @ G @ a_star
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
result = differential_evolution(objective_function, bounds, constraints=[nonlinear_constraint])

def compute_a_star(t, p, G):
    n = len(t)
    I = np.eye(n)
    T_sqrt = np.diag(np.sqrt(t))
    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ np.ones(n)
    return a_star

optimal_t = result.x  # This assumes 'result' is the output of your differential_evolution call
a_star_optimal = compute_a_star(optimal_t, p, G)

print("Computed a_star using optimal t:", a_star_optimal)
print("Optimal t:", result.x)
print("Minimum objective value:", result.fun)
