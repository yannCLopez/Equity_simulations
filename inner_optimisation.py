
#INNER OPTIMISATION PROBLEM. 

import numpy as np

# Fixed parameters
G = np.array([
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0]
])
p = 0.7
t = [0.61812435, 0.24963716, 0.53499259, 0.62381142, 0.43483632]  # Example t vector, this will be fixed for now but optimized later

# Objective function
def f(a, t):
    diag_t_sqrt = np.diag(np.sqrt(t))
    term = p * np.linalg.inv(np.eye(5) - p * diag_t_sqrt @ G) @ diag_t_sqrt @ np.ones(5)
    return np.linalg.norm(a - term)**2

# Gradient of the objective function with respect to a
def grad_f(a, t):
    diag_t_sqrt = np.diag(np.sqrt(t))
    term = p * np.linalg.inv(np.eye(5) - p * diag_t_sqrt @ G) @ diag_t_sqrt @ np.ones(5)
    #rev
    return 2 * (a - term)

# Gradient descent
def gradient_descent(t, learning_rate=0.01, num_iterations=1000, num_starts=300):
    #rev
    n = 5
    best_a = None
    #rev
    best_f = np.inf
    
    for _ in range(num_starts):
        a = np.random.uniform(0, 1000, n)
        for _ in range(num_iterations):
            #rev
            a -= learning_rate * grad_f(a, t)

            #rev
            current_f = f(a, t)
            if current_f < best_f:
                best_f = current_f
                best_a = a
    
    return best_a, best_f

# Example usage (with a random t for now)
best_a, best_f = gradient_descent(t)
print("Best a:", best_a)
print("Best objective value:", best_f)
