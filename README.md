# Equity_simulations
## Main
  - **global_optimisation**: Performs the optimization of t using differential_evolution with the specified objective function and constraint, where bounds limits each component of t between 0 and 1.
  - **simulation_pearson.py**: Graph Generation (generate_G): Generates connected graphs G with nodes connected by edges with randomly assigned weights.
Optimization: Uses differential_evolution to optimize a vector t against the objective function objective_function, which depends on t, the constant p, and the graph G.
Normalization and Condition Checking (check_condition): Normalizes the optimized vector t and a derived vector a_star, ensuring the elements of each vector differ by at least 0.001.
Correlation Calculation (pearsonr): Computes Pearson correlations between t and a_star, both in their original and normalized forms.
Data Recording: Logs results, including matrices of G, normalized vectors, and statistical metrics like Pearson correlation, standard deviations, and others into a CSV file file_path.
  - **simulation_polynomial.py**: Derivative Machine: It dynamically generates a random polynomial with multiple variables and computes its derivatives. The polynomial's coefficients and degree are randomly determined within specified limits.
Optimization Machine: It performs an optimization to find variable values that minimize a derived objective function, which incorporates the polynomial's derivatives. This process is repeated multiple times to determine the best solution based on a "principal surplus" metric, which evaluates the polynomial's effectiveness under the optimal conditions found. The solution with the highest principal surplus is selected and displayed.
## Plots_Finance_Presentation/core_periphery_network:
Contains :
  - Figure 2_1 to Figure_5_2. These plot sigma and mu against kappa, for a core node and a periphery node in a core periphery network, when we vary their b. The range of kappas for both nodes overlaps.
  - **core_graph_kappa.py**: The code used to create the above
  - variable_values_figure_3.tex, output_4-5.tex: Files containing the values taken by the relevant variables in the corresponding figures.
  - graph.png: a depiction of the corresponding core periphery graph]
  - **core_graph_figure.py**: the code used to draw the above.
  - **optimisation_slope.py**: First shot at attempting to find G and b for maximum difference in slopes between central and periphery nodes. 
