######################################################################################################
# M-estimation for Fusion Designs
#       Example 2: transport Y with measurement error
#
# Paul Zivich (2022/6/13)
######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

from dgm import generate_data_ex2


# Defining the estimating functions


def psi_ignore1(theta):
    # Calculating mean of Y*, ignoring measurement error from S=1
    return ys1 - theta


def psi_ignore2(theta):
    # Calculating mean of Y from S=2
    return ys2 - theta


def psi_fusion(theta):
    # Measurement values (sensitivity and 1-specificity)
    sens = (s == 2) * (y == 1) * (y_star - theta[0])
    spec = (s == 2) * (y == 0) * ((1 - y_star) - theta[1])

    s_model = ee_regression(theta[4:], X=W, y=p, model='logistic')
    s_model = np.nan_to_num(s_model, copy=False, nan=0.)
    pi_s = inverse_logit(np.dot(W, theta[4:]))
    weight = pi_s / (1 - pi_s)

    # Inverse odds weighted mean of Y*
    omega = np.where(s == 1, weight * (y_star - theta[2]), 0)
    # Rogan & Gladden 1978 estimator
    mu = np.ones(s.shape)*theta[3]*(theta[0] + theta[1] - 1) - (theta[2] + theta[1] - 1)

    # Returning the stack of estimating functions
    return np.vstack([sens[None, :],
                      spec[None, :],
                      omega[None, :],
                      mu[None, :],
                      s_model])


# Simulation study
reps = 2000
np.random.seed(13659321)

# Simulating true value of E[Y | S=1]
tdat = generate_data_ex2(n1=1, n2=1, n3=1000000, missing=False)
truth = np.mean(tdat.loc[tdat['S'] == 3, 'Y'])
print(truth)

# Storage for results
bias_s1, cover_s1 = [], []
bias_s2, cover_s2 = [], []
bias_f, cover_f = [], []

for i in range(reps):
    # Generating data
    d = generate_data_ex2(n1=750, n2=200, n3=1500, missing=True)
    d['C'] = 1

    # Organizing the data for input into the estimating functions
    ys1 = np.asarray(d.loc[d['S'] == 1, 'Y*'])
    ys2 = np.asarray(d.loc[d['S'] == 2, 'Y'])
    s = np.asarray(d['S'])
    W = np.asarray(d[['C', 'W']])
    y = np.asarray(d['Y'])
    y_star = np.asarray(d['Y*'])
    p = np.where(d['S'] == 3, 1, np.nan)
    p = np.where(d['S'] == 1, 0, p)

    # Approach 1: ignore measurement error and use S=1
    mest = MEstimator(psi_ignore1, init=[0.5])
    mest.estimate(solver='lm')
    est = mest.theta[0]
    ci = mest.confidence_intervals()[0]
    bias_s1.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_s1.append(1)
    else:
        cover_s1.append(0)

    # Approach 2: use S=2
    mest = MEstimator(psi_ignore2, init=[0.5])
    mest.estimate(solver='lm')
    est = mest.theta[0]
    ci = mest.confidence_intervals()[0]
    bias_s2.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_s2.append(1)
    else:
        cover_s2.append(0)

    # Approach 2: fusion design
    mest = MEstimator(psi_fusion, init=[0.5, 0.5, 0.5, 0.5, 0., 0.])
    mest.estimate(solver='lm')
    est = mest.theta[3]
    ci = mest.confidence_intervals()[3]
    bias_f.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_f.append(1)
    else:
        cover_f.append(0)


fig, ax = plt.subplots(figsize=(6.75, 4.5))
# Bias results
ax.hlines([0], [0], [4], colors='gray', linestyles='--', zorder=1)
parts = ax.violinplot([bias_s1, bias_s2, bias_f], positions=[1, 2, 3], showmeans=True, widths=0.25)
parts['bodies'][0].set_zorder(2)
for pc in parts['bodies']:
    pc.set_color("blue")
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
    vp = parts[partname]
    vp.set_edgecolor('blue')
# Confidence interval results
ax2 = ax.twinx()
ax2.hlines([0.95], [0], [5], colors='gray', linestyles=':', zorder=1)
ax2.plot([1, 2, 3], [np.mean(cover_s1), np.mean(cover_s2), np.mean(cover_f)],
         'D', color='mediumblue', markersize=7, zorder=3)
# Making plot look nice
plt.xticks([1, 2, 3], [r"Use $S=1$", r"Use $S=2$", "Fusion design"])
plt.xlim([0.5, 3.5])
ax.set_ylim([-0.25, 0.25])
ax.set_ylabel(r"Bias ($\hat{\mu} - \mu$)")
ax2.set_ylim([0, 1])
ax2.set_ylabel("95% CI Coverage")
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("../images/sim_a2_results.png", format='png', dpi=300)
plt.close()
