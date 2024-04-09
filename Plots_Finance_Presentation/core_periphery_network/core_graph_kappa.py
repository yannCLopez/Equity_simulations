# Program: To plot the equity of an agent in a core periphery network versus \kappa.
# References: Building on Anant's code for a star graph
# Date: 4-9-2024

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt

# Parameters
num_agents = 12

G = np.array([
    [0, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [3, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [3, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
p = 0.2
a = 0.575 # we set all entries of b_base_central equal to a. We also scale the changes in b_base_periphery by a. 

num_iter = 5 # Number of data points we want for each type of node
rand_bound = 0.1 # The distance from an individual productivity of 1. Essentially b is sampled from [1-rand_bound,1+rand_bound]
central_equity = np.zeros(num_iter) # Store the equity of the central node as its b is varied, keeping the others constant
peripheral_equity = np.zeros(num_iter) # Store the equity of a peripheral node as its b is varied, keeping the others constant



b0_list = np.random.uniform(-rand_bound,rand_bound,num_iter) # Vary the 

b0_list = np.sort(b0_list)
kappa_axis_central = np.zeros(num_iter)
kappa_axis_periphery = np.zeros(num_iter)
optimal_t_periphery = np.zeros(num_iter)
optimal_t_central = np.zeros(num_iter)


for iter in range(num_iter):
	print(iter)
	#b = 1 + np.random.uniform(-0.2, 0.2, num_agents)
	b_base_central = np.full(num_agents, a)
	print (b0_list[iter])
	print (b_base_central[0])
	b_base_central[0] = b_base_central[0] + b0_list[iter] # Individual production coefficients of each agent
	b = b_base_central 
	print (b)
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
	a_star_optimal_central = compute_a_star(optimal_t, p, G, b)

	kappa_all_agents = b + G @ a_star_optimal_central
	#r
	kappa_axis_central[iter] = kappa_all_agents[0]
	optimal_t_central[iter] = optimal_t[0]


for iter in range(num_iter):
	print(iter)
	b_base_periphery = np.ones(num_agents)
	b_base_periphery[6] = b_base_periphery[6] + b0_list[iter]/a # The individual coefficients of each agent
	b = b_base_periphery 
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
	peripheral_equity[iter] = 2*np.sqrt(optimal_t[6])
	a_star_optimal_periphery = compute_a_star(optimal_t, p, G, b)

	kappa_all_agents = b + G @ a_star_optimal_periphery
	kappa_axis_periphery[iter] = kappa_all_agents[6]
	optimal_t_periphery[iter] = optimal_t[6]

# MU
# Compute the line of best fit for the peripheral nodes (the blue dots)
m_peripheral, b_peripheral = np.polyfit(kappa_axis_periphery,peripheral_equity,1)
print(m_peripheral)
# Compute the linw of best fit for the central node (the red dots)
m_central, b_central = np.polyfit(kappa_axis_central,central_equity,1)

print(m_central)

# SIGMA
# Compute the line of best fit for the peripheral nodes (the blue dots)
m_peripheral_sigma, b_peripheral_sigma = np.polyfit(kappa_axis_periphery,optimal_t_periphery,1)
print(m_peripheral_sigma)

# Compute the linw of best fit for the central node (the red dots)
m_central_sigma, b_central_sigma = np.polyfit(kappa_axis_central,optimal_t_central,1)

print(m_central_sigma)

#Store variables' values
with open('output.tex', 'w') as f:
    f.write("\\documentclass{article}\n")
    f.write("\\begin{document}\n")
    f.write("\\section{Variables}\n")
    f.write("\\begin{enumerate}\n")
    f.write(f"\\item $b0\_list$: {b0_list}\n")
    f.write(f"\\item $b\_base\_central$: {b_base_central}\n")
    f.write(f"\\item $b\_base\_periphery$: {b_base_periphery}\n")
    f.write(f"\\item $p$: {p}\n")
    f.write(f"\\item $G$: {G}\n")
    f.write(f"\\item $a\_star\_optimal\_central$: {a_star_optimal_central}\n")
    f.write(f"\\item $a\_star\_optimal\_periphery$: {a_star_optimal_periphery}\n")
    f.write(f"\\item $optimal\_t\_central$: {optimal_t_central}\n")
    f.write(f"\\item $optimal\_t\_periphery$: {optimal_t_periphery}\n")
    f.write(f"\\item $kappa\_axis\_central$: {kappa_axis_central}\n")
    f.write(f"\\item $kappa\_axis\_periphery$: {kappa_axis_periphery}\n")
    f.write("\\end{enumerate}\n")
    f.write("\\end{document}\n")

#Set position of slope labels
x_range = np.max(kappa_axis_central) - np.min(kappa_axis_central)
y_range = np.max(optimal_t_central) - np.min(optimal_t_central)
x_pos_central = np.max(kappa_axis_central) - 0.2 * x_range
y_pos_central = np.max(optimal_t_central) - 1.9 * y_range

x_range_periphery = np.max(kappa_axis_periphery) - np.min(kappa_axis_periphery)
y_range_periphery = np.max(optimal_t_periphery) - np.min(optimal_t_periphery)
x_pos_periphery = np.max(kappa_axis_periphery) - 0.2 * x_range_periphery
y_pos_periphery = np.min(optimal_t_periphery) + 5.3 * y_range_periphery

y_range_mu = np.max(central_equity) - np.min(central_equity)
y_pos_central_mu = np.max(central_equity) - 2.5 * y_range_mu

y_range_periphery_mu = np.max(peripheral_equity) - np.min(peripheral_equity)
y_pos_periphery_mu = np.min(peripheral_equity) + 3.8 * y_range_periphery_mu

print (f'optimal actions (central) = {a_star_optimal_central}')
print (f'optimal actions (periphery) = {a_star_optimal_periphery}')

######## PLOTS FOR MU ########
######## Plots for the central node #########

fig, ax = plt.subplots()

plt.plot(kappa_axis_central,central_equity,'ro')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.legend(['Central node'],loc='upper left')
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)
plt.show()

fig, ax = plt.subplots()

# Now plot the points with the best fit line
plt.plot(kappa_axis_central,central_equity,'ro')
plt.plot(kappa_axis_central,m_central*kappa_axis_central + b_central,'r--')
plt.legend([r'Central node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_central, y_pos_central_mu,r'slope=%.2f'%m_central,bbox={'facecolor':'red','alpha':0.5,'pad':10})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)
plt.show()

######## Plots for the peripheral node ######

fig, ax = plt.subplots()

plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
plt.legend(['Peripheral node'],loc='upper left')
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()

fig, ax = plt.subplots()

plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.plot(kappa_axis_periphery,m_peripheral*kappa_axis_periphery + b_peripheral,'b--')
plt.legend(['Peripheral node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_periphery, y_pos_periphery_mu,r'slope=%.2f'%m_peripheral,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()

#### Plot them combined

fig, ax = plt.subplots()

plt.plot(kappa_axis_central,central_equity,'ro')
plt.plot(kappa_axis_periphery,peripheral_equity,'bo')
plt.plot(kappa_axis_central,m_central*kappa_axis_central + b_central,'r--')
plt.plot(kappa_axis_periphery,m_peripheral*kappa_axis_periphery + b_peripheral,'b--')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\mu$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_central, y_pos_central_mu,r'slope=%.2f'%m_central,bbox={'facecolor':'red','alpha':0.5,'pad':10})
plt.text(x_pos_periphery, y_pos_periphery_mu,r'slope=%.2f'%m_peripheral,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
plt.legend(['Central node','Peripheral node'],loc='upper left', bbox_to_anchor=(0, 1.15))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()



######## PLOTS FOR SIGMA ########
######## Plots for the central node #########


fig, ax = plt.subplots()

plt.plot(kappa_axis_central,optimal_t_central,'ro')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\sigma$', fontsize=10)
plt.legend(['Central node'],loc='upper left')
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)
plt.show()

fig, ax = plt.subplots()

# Now plot the points with the best fit line
plt.plot(kappa_axis_central,optimal_t_central,'ro')
plt.plot(kappa_axis_central,m_central_sigma*kappa_axis_central + b_central_sigma,'r--')
plt.legend([r'Central node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\sigma$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_central, y_pos_central,r'slope=%.2f'%m_central_sigma,bbox={'facecolor':'red','alpha':0.5,'pad':10})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)
plt.show()

######## Plots for the peripheral node ######

fig, ax = plt.subplots()

plt.plot(kappa_axis_periphery,optimal_t_periphery,'bo')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\sigma$', fontsize=10)
plt.legend(['Peripheral node'],loc='upper left')
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()

fig, ax = plt.subplots()

plt.plot(kappa_axis_periphery,optimal_t_periphery,'bo')
plt.plot(kappa_axis_periphery,m_peripheral_sigma*kappa_axis_periphery + b_peripheral_sigma,'b--')
plt.legend(['Peripheral node'],loc='upper left')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\sigma$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_periphery, y_pos_periphery,r'slope=%.2f'%m_peripheral_sigma,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()

#### Plot them combined

fig, ax = plt.subplots()

plt.plot(kappa_axis_central,optimal_t_central,'ro')
plt.plot(kappa_axis_periphery,optimal_t_periphery,'bo')
plt.plot(kappa_axis_central,m_central_sigma*kappa_axis_central + b_central_sigma,'r--')
plt.plot(kappa_axis_periphery,m_peripheral_sigma*kappa_axis_periphery + b_peripheral_sigma,'b--')
plt.xlabel(r'$\kappa$', fontsize=10)
plt.ylabel(r'$\sigma$', fontsize=10)
#plt.xlim(0.95,1.4)
#plt.ylim(0.35,0.75)
plt.text(x_pos_central, y_pos_central,r'slope=%.2f'%m_central_sigma,bbox={'facecolor':'red','alpha':0.5,'pad':10})
plt.text(x_pos_periphery, y_pos_periphery,r'slope=%.2f'%m_peripheral_sigma,bbox={'facecolor':'blue','alpha':0.5,'pad':10})
plt.legend(['Central node','Peripheral node'],loc='upper left', bbox_to_anchor=(0, 1.15))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False,bottom=False)

plt.show()




