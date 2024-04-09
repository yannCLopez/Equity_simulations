import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Define the adjacency matrix
G_adj = np.array([
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
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

import numpy as np



# Create a graph from the adjacency matrix
G = nx.Graph()
for i in range(len(G_adj)):
    for j in range(i+1, len(G_adj)):
        if G_adj[i][j] > 0:
            G.add_edge(i, j, weight=G_adj[i][j])

# Manually set positions for all nodes
pos = {
    0: (0, 1),
    1: (-1, 0),
    2: (1, 0),
    3: (0, 2),
    4: (-1, 2),
    5: (1, 2),
    6: (-2, 0.5),
    7: (-2, -0.5),
    8: (-2, 0),
    9: (2, 0),
    10: (2, -0.5),
    11: (2, 0.5)
}

# Draw the graph
plt.figure(figsize=(10, 10))
# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)

# Draw edges with thickness proportional to weights
edges = G.edges(data=True)
#weights = [G[u][v]['weight'] for u, v, attr in edges]
weights = [attr['weight'] for u, v, attr in edges]

nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12)

plt.axis('on')  # Turn off the axis
plt.grid(False)
plt.show()

