######################################################################################################
# M-estimation for Fusion Designs
#       Example 1: measurement error
#
# Paul Zivich (2022/6/13)
######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from delicatessen import MEstimator

from dgm import generate_data_ex1


# Defining the estimating functions


def psi_ignore(theta):
    # Calculating mean of Y*, ignoring measurement error
    return yr - theta


def psi_fusion(theta):
    # Measurement values (sensitivity and 1-specificity)
    sens = (s == 2) * (y == 1) * (y_star - theta[0])
    spec = (s == 2) * (y == 0) * ((1 - y_star) - theta[1])

    # Mean of Y*
    omega = (s == 1) * (y_star - theta[2])

    # Rogan & Gladden 1978 estimator
    mu = np.ones(s.shape) * theta[3] * (theta[0] + theta[1] - 1) - (theta[2] + theta[1] - 1)

    # Returning the stack of estimating functions
    return np.vstack([sens[None, :],
                      spec[None, :],
                      omega[None, :],
                      mu[None, :]])


# Simulation study
reps = 2000                   # Setting number of sims
np.random.seed(13659321)      # Setting seed for replication

# Simulating true value of E[Y | S=1]
tdat = generate_data_ex1(n1=1000000, n2=1, missing=False)    # Generating true value of Y
truth = np.mean(tdat.loc[tdat['S'] == 1, 'Y'])               # Calculating \mu
print(truth)                                                 # Print \mu to the console

# Storage for results
bias_n, cover_n = [], []     # Storage for results
bias_f, cover_f = [], []     # Storage for results

for i in range(reps):
    # Generating data
    d = generate_data_ex1(n1=750, n2=200, missing=True)      # Generating observed data

    # Organizing the data for input into the estimating functions
    s = np.asarray(d['S'])                                   # Extract S indicator as np.array
    y = np.asarray(d['Y'])                                   # Extract Y as np.array
    y_star = np.asarray(d['Y*'])                             # Extract Y* as np.array
    yr = np.asarray(d.loc[d['S'] == 1, 'Y*'])                # Extract Y* only for S=1 as np.array

    # Approach 1: ignore measurement error
    mest = MEstimator(psi_ignore, init=[0.5])                # Setup M-estimator
    mest.estimate(solver='lm')                               # Solve the M-estimator
    est = mest.theta[0]                                      # Extract point-estimate
    ci = mest.confidence_intervals()[0]                      # Extract confidence intervals
    bias_n.append(est - truth)                               # Calculate bias and store
    if ci[0] < truth < ci[1]:                                # Calculate coverage (yes or no) and store
        cover_n.append(1)
    else:
        cover_n.append(0)

    # Approach 2: fusion design
    mest = MEstimator(psi_fusion, init=[0.75, 0.75, 0.5, 0.5])  # Setup M-estimator
    mest.estimate(solver='lm')                                  # Solve the M-estimator
    est = mest.theta[-1]                                        # Extract point-estimate
    ci = mest.confidence_intervals()[-1]                        # Extract confidence intervals
    bias_f.append(est - truth)                                  # Calculate bias and store
    if ci[0] < truth < ci[1]:                                   # Calculate coverage (yes or no) and store
        cover_f.append(1)
    else:
        cover_f.append(0)


# Creating plot of results for the slides
fig, ax = plt.subplots(figsize=(6.75, 4.5))

# Bias results
ax.hlines([0], [0], [3], colors='gray',                    # Reference line at 0 bias
          linestyles='--', zorder=1)
parts = ax.violinplot([bias_n, bias_f], positions=[1, 2],  # Violinplot of results
                      showmeans=True, widths=0.25)
parts['bodies'][0].set_zorder(2)                           # Have violinplot in front of the reference line
for pc in parts['bodies']:                                 # Changing colors of the violinplot
    pc.set_color("blue")
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):    # Changing colors of the violinplot
    vp = parts[partname]
    vp.set_edgecolor('blue')

# Confidence interval results
ax2 = ax.twinx()                                           # Copy X-axis but use a new right y-axis
ax2.hlines([0.95], [0], [5], colors='gray',                # Reference line at 95% coverage
           linestyles=':', zorder=1)
ax2.plot([1, 2], [np.mean(cover_n), np.mean(cover_f)],     # Plot coverage as points
         'D', color='mediumblue', markersize=7, zorder=3)

# Making plot look nice
plt.xticks([1, 2], ["Ignore measurement\nerror", "Fusion design"])
plt.xlim([0.5, 2.5])
ax.set_ylim([-0.25, 0.25])
ax.set_ylabel(r"Bias ($\hat{\mu} - \mu$)")
ax2.set_ylim([0, 1])
ax2.set_ylabel("95% CI Coverage")
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Final format and saving plot
plt.tight_layout()
plt.savefig("../images/sim_a1_results.png", format='png', dpi=300)
plt.close()
