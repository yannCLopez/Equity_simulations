import numpy as np
 

# Given fixed parameters
G = np.array([
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0]
])
p = 0.7
n = 5

# Function to compute kappa, U, and u_i' based on the given definitions
def compute_kappa_values(sigma, a_star):
    # Calculate the gradient of Y
    grad_Y = 1 + G @ a_star

    # Since H is identity, kappa is simply grad_Y
    kappa = grad_Y

    # Compute the U matrix
    Y = 1 @ a_star + 0.5 * a_star @ G @ a_star
    U = np.diag(p * Y * np.sqrt(sigma) - 0.5 * a_star**2)

    # Compute the inverse term for kappa_bar
    kappa_bar = kappa @ np.linalg.inv(np.eye(n) - U @ G)

    # Calculate u_i' for each sigma_i
    u_i_prime = p * (1 @ a_star + 0.5 * a_star @ G @ a_star) * 0.5 / np.sqrt(sigma)

    # Compute the product kappa_i * kappa_bar_i * u_i_prime_i
    result = kappa * kappa_bar * u_i_prime

    return result

# Example usage
sigma_example = np.random.rand(n)  # Example contract vector
a_star_example = np.random.rand(n)  # Example optimal action vector
result = compute_kappa_values(sigma_example, a_star_example)
print(result)


####################################################
#INNER OPTIMISATION PROBLEM. 


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
