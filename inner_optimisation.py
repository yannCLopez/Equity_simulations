import numpy as np
from scipy.optimize import minimize
import time
start_time = time.time()  # Start the timer


# Economic parameters
p = 0.7

# Example t vector, this will be fixed for now but optimized later
t = np.array([0.61812435, 0.24963716, 0.53499259, 0.62381142, 0.43483632])

# Objective function
def f(a, t, G, p):
    diag_t_sqrt = np.diag(np.sqrt(t))
    term = p * np.linalg.inv(np.eye(len(t)) - p * diag_t_sqrt @ G) @ diag_t_sqrt @ np.ones(len(t))
    return np.linalg.norm(a - term)**2

# Gradient of the objective function with respect to a
def grad_f(a, t, G, p):
    diag_t_sqrt = np.diag(np.sqrt(t))
    term = p * np.linalg.inv(np.eye(len(t)) - p * diag_t_sqrt @ G) @ diag_t_sqrt @ np.ones(len(t))
    return 2 * (a - term)

# Gradient descent using scipy.optimize.minimize
def gradient_descent(t, G, p, num_starts=300):
    best_a = None
    best_f = np.inf
    
    bounds = [(0, None) for _ in range(len(t))]  # Constraints: a_i >= 0
    for _ in range(num_starts):
        # Initial random guess for starting the optimization
        initial_guess = np.random.uniform(0, 100, len(t))
        
        # Optimization using scipy.optimize's minimize function
        result = minimize(f, initial_guess, args=(t, G, p), jac=grad_f, bounds=bounds, method='L-BFGS-B')

        # Check if the result from this start point is better than the best found so far
        if result.fun < best_f:
            best_f = result.fun
            best_a = result.x
    
    return best_a, best_f

# Matrix G and other parameters should be provided as inputs
G = np.array([
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0]
])



best_a, best_f = gradient_descent(t, G, p)

end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time
print (elapsed_time)
print("Best a:", best_a)
print("Best objective value:", best_f)
