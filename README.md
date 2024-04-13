# Equity_simulations
## Main
  - simulation_polynomial.py: Derivative Machine: It dynamically generates a random polynomial with multiple variables and computes its derivatives. The polynomial's coefficients and degree are randomly determined within specified limits.
Optimization Machine: It performs an optimization to find variable values that minimize a derived objective function, which incorporates the polynomial's derivatives. This process is repeated multiple times to determine the best solution based on a "principal surplus" metric, which evaluates the polynomial's effectiveness under the optimal conditions found. The solution with the highest principal surplus is selected and displayed.
## Plots_Finance_Presentation/core_periphery_network:
Contains :
  - Figure 2_1 to Figure_5_2. These plot sigma and mu against kappa, for a core node and a periphery node in a core periphery network, when we vary their b. The range of kappas for both nodes overlaps.
  - core_graph_kappa.py: The code used to create the above
  - variable_values_figure_3.tex, output_4-5.tex: Files containing the values taken by the relevant variables in the corresponding figures.
  - graph.png: a depiction of the corresponding core periphery graph]
  - core_graph_figure.py: the code used to draw the above.
  - optimisation_slope.py: First shot at attempting to find G and b for maximum difference in slopes between central and periphery nodes. 
