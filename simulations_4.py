import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt

num_agents = 6
p=0.2
#G = np.array([[0, 1, 1, 0, 1],[1, 0, 0, 1, 1],[1, 0, 0, 1, 0],[0, 1, 1, 0, 1],[1, 1, 0, 1, 0]])
#G=np.array([[0.1, 0.02, 0.03], [0.02, 0.2, 0.03], [0.03, 0.03, 0.3]])

G = np.zeros((num_agents,num_agents))
# Create a star graph wih 1 central node and (n-1) peripheral nodes
for i in range(num_agents):
	if i==0:
		G[i,1:num_agents] = np.ones(num_agents-1)
	else:
		G[i,0] = 1 

print(G)

num_iter = 1000
b0_vary = np.linspace(0.1,2,num_iter)
central_equity_ratio = np.zeros(num_iter)
peripheral_equity_ratio_to_central = np.zeros(num_iter)
peripheral_equity_ratio_to_peripheral = np.zeros(num_iter)

for iter in range(num_iter):
	print(iter)
	b = np.ones(num_agents)
	b[0] = b0_vary[iter]

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
	result = differential_evolution(objective_function, bounds, constraints=[nonlinear_constraint])

	def compute_a_star(t, p, G, b):
	    n = len(t)
	    I = np.eye(n)
	    T_sqrt = np.diag(np.sqrt(t))
	    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ b
	    return a_star

	optimal_t = result.x  # This assumes 'result' is the output of your differential_evolution call
	a_star_optimal = compute_a_star(optimal_t, p, G, b)

	central_equity_ratio[iter] = optimal_t[0]/optimal_t[1]

	#print("Computed a_star using optimal t:", a_star_optimal)
	#print("Optimal t:", result.x)
	#print("Minimum objective value:", result.fun)

for iter in range(num_iter):
	print(iter)
	b = np.ones(num_agents)
	b[1] = b0_vary[iter]

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
	result = differential_evolution(objective_function, bounds, constraints=[nonlinear_constraint])

	def compute_a_star(t, p, G, b):
	    n = len(t)
	    I = np.eye(n)
	    T_sqrt = np.diag(np.sqrt(t))
	    a_star = p * np.linalg.inv(I - p * T_sqrt @ G) @ T_sqrt @ b
	    return a_star

	optimal_t = result.x  # This assumes 'result' is the output of your differential_evolution call
	a_star_optimal = compute_a_star(optimal_t, p, G, b)

	peripheral_equity_ratio_to_central[iter] = optimal_t[1]/optimal_t[0]
	peripheral_equity_ratio_to_peripheral[iter] = optimal_t[1]/optimal_t[2]

	#print("Computed a_star using optimal t:", a_star_optimal)
	#print("Optimal t:", result.x)
	#print("Minimum objective value:", result.fun)

plt.plot(b0_vary,central_equity_ratio)
plt.plot(b0_vary,peripheral_equity_ratio_to_central)
plt.plot(b0_vary,peripheral_equity_ratio_to_peripheral)
plt.plot(b0_vary,np.ones(num_iter),'r--')
plt.xlabel(r"Linear coefficient")
plt.legend(['Central node','Peripheral to central node','Peripheral to peripheral node'])
plt.ylabel(r"Equity ratio")
plt.show()
