import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import mossspider
from mossspider import NetworkTMLE
from mossspider.dgm import uniform_network, generate_observed

#####################################
# Creating data

# Generating graph
G = uniform_network(n=500, degree=[1, 4],
                    seed=202205)

# Selecting out the largest component
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = nx.convert_node_labels_to_integers(G.subgraph(Gcc[0]))

# Simulating covariates for largest component
H = generate_observed(G0,
                      seed=202205)

# Determine position for nodes
pos = nx.spring_layout(H, seed=202205)

# Color nodes by A
color_map = []                         # Empty list for colors
for node, data in H.nodes(data=True):  # For loop over nodes
    if data['A'] == 1:                 # If A=1
        color_map.append('blue')       # ... set color to blue
    else:                              # If A=0
        color_map.append('red')        # ... set color to green

# Drawing network
nx.draw(H,
        node_color=color_map,
        node_size=20,
        pos=pos,
       )
plt.savefig("images/example_network.png", format='png', dpi=300)
plt.close()

#####################################
# Running Analysis

# Initialize NetworkTMLE
ntmle = NetworkTMLE(network=H,
                    exposure="A",
                    outcome="Y")
# Model for Pr(A | W, W^s; \delta)
ntmle.exposure_model(model="W + W_sum")
# Model for Pr(A^s | A, W, W^s; \gamma)
ntmle.exposure_map_model(model="A + W + W_sum",
                         measure="sum",
                         distribution="poisson")
# Model for E[Y | A, A^s, W, W^s; \alpha]
ntmle.outcome_model(model="A + A_sum + W + W_sum")

# Estimation
point = []                    # Storage for psi
lcl_d, ucl_d = [], []         # Storage for CI direct
lcl_l, ucl_l = [], []         # Storage for CI latent

# Policies to evaluate
policy = [0.1, 0.2, 0.3, 0.4, 0.5,
          0.6, 0.7, 0.8, 0.9]

# Evaluating each policy
for p in policy:
    ntmle.fit(p=p,            # Policy
              samples=200,    # replicates
              seed=20220316)  # random seed
    # Saving output
    cid = ntmle.conditional_ci
    cil = ntmle.conditional_latent_ci
    point.append(ntmle.marginal_outcome)
    lcl_d.append(cid[0])
    lcl_l.append(cil[0])
    ucl_d.append(cid[1])
    ucl_l.append(cil[1])

plt.plot(policy, point, 'o-', color='blue')
plt.fill_between(policy, lcl_d, ucl_d, color='blue', alpha=0.2)
plt.fill_between(policy, lcl_l, ucl_l, color='gray', alpha=0.2)
plt.xlabel(r"Policy of Interest ($\Pr^*(A_i=1)=\omega$)")
plt.ylabel(r"$\mathbf{W}$-conditional causal mean ($\hat{\psi}$)")
plt.savefig("images/example_results.png", format='png', dpi=300)
plt.close()
