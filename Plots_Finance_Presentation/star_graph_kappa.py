# Program: To plot the equity of an agent in a star graph versus \kappa.
# References: Building on Yann's code for a core-periphery network
# Date: 4-8-2024

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt

num_agents = 6
p = 0.2

num_iter = 10 # Number of data points we want for each type of node
rand_bound = 0.1 # The distance from an individual productivity of 1. Essentially b is sampled from [1-rand_bound,1+rand_bound]
central_equity = np.zeros(num_iter) # Store the equity of the central node as its b is varied, keeping the others constant
peripheral_equity = np.zeros(num_iter) # Store the equity of a peripheral node as its b is varied, keeping the others constant


G = np.zeros((num_agents,num_agents))
### Create a star graph wih 1 central node and (n-1) peripheral nodes
for i in range(num_agents):
	if i==0:
		G[i,1:num_agents] = np.ones(num_agents-1)
	else:
		G[i,0] = 1 

print(G)

b0_list = np.random.uniform(-rand_bound,rand_bound,num_iter) # Vary the 

b0_list = np.sort(b0_list)
kappa_axis_central = np.zeros(num_iter)
kappa_axis_periphery = np.zeros(num_iter)

for iter in range(num_iter):
	print(iter)
	#b = 1 + np.random.uniform(-0.2, 0.2, num_agents)
	b_base = np.ones(num_agents)
	b_base[0] = b_base[0] + b0_list[iter] # Individual production coefficients of each agent
	b = b_base 
	# Could also write as: 
	# b = np.ones(num_agents)
	# b[0] = b[0] + b0_list[iter]	

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

	optimal_t = (result.x)  # This assumes 'result' is the output of your differential_evolution call
	central_equity[iter] = 2*np.sqrt(optimal_t[0])
	a_star_optimal = compute_a_star(optimal_t, p, G, b)

	kappa_all_agents = b + G @ a_star_optimal
	kappa_axis_central[iter] = kappa_all_agents[0]

for iter in range(num_iter):
	print(iter)
	b_base = np.ones(num_agents)
	b_base[1] = b_base[1] + b0_list[iter] # The individual coefficients of each agent
	b = b_base 
	# Could also write as: 
	# b = np.ones(num_agents)
	# b[0] = b[0] + b0_list[iter]	

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

	optimal_t = (result.x)  # This assumes 'result' is the output of your differential_evolution call
	peripheral_equity[iter] = 2*np.sqrt(optimal_t[1])
	a_star_optimal = compute_a_star(optimal_t, p, G, b)

	kappa_all_agents = b + G @ a_star_optimal
	kappa_axis_periphery[iter] = kappa_all_agents[1]

# Compute the line of best fit for the peripheral nodes (the blue dots)
m_peripheral, b_peripheral = np.polyfit(kappa_axis_periphery,peripheral_equity,1)
print(m_peripheral)
# Compute the linw of best fit for the central node (the red dots)
m_central, b_central = np.polyfit(kappa_axis_central,central_equity,1)

print(m_central)

######## Plots for the central node #########

plt.plot(kappa_axis_central,central_equity,'ro')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.legend(['Central node'],loc='upper left')
plt.xlim(0.95,1.4)
plt.ylim(0.35,0.75)
plt.show()

# Now plot the points with the best fit line
plt.plot(kappa_axis_central,central_equity,'ro')
plt.plot(kappa_axis_central,m_central*kappa_axis_central + b_central,'r--')
plt.legend([r'Central node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.xlim(0.95,1.4)
plt.ylim(0.35,0.75)
plt.text(1.30,0.65,r'slope=%.2f'%m_central,bbox={'facecolor':'red','alpha':0.5,'pad':10})
plt.show()

######## Plots for the peripheral node ######

plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.legend(['Peripheral node'],loc='upper left')
plt.xlim(0.95,1.4)
plt.ylim(0.35,0.75)
plt.show()

plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.plot(kappa_axis_periphery,m_peripheral*kappa_axis_periphery + b_peripheral,'b--')
plt.legend(['Peripheral node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.xlim(0.95,1.4)
plt.ylim(0.35,0.75)
plt.text(1.10,0.4,r'slope=%.2f'%m_peripheral,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
plt.show()

#### Plot them combined

plt.plot(kappa_axis_central,central_equity,'ro')
plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.plot(kappa_axis_central,m_central*kappa_axis_central + b_central,'r--')
plt.plot(kappa_axis_periphery,m_peripheral*kappa_axis_periphery + b_peripheral,'b--')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.xlim(0.95,1.4)
plt.ylim(0.35,0.75)
plt.text(1.3,0.65,r'slope=%.2f'%m_central,bbox={'facecolor':'red','alpha':0.5,'pad':10})
plt.text(1.10,0.4,r'slope=%.2f'%m_peripheral,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
plt.legend(['Central node','Peripheral node'],loc='upper left')
plt.show()
