import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_logistic_regression
from delicatessen.utilities import inverse_logit

from dgm import generate_data_1, generate_x, generate_data, generate_data_2, generate_data_3

np.random.seed(13659321)

# Creating generic W,V distribution plot
width = 0.30       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar([1, 2, 3],
       [0.25, 0.75, 0.65],
       width, color='blue', edgecolor='k', label=r'$W$')
ax.bar([1.3, 2.3, 3.3],
       [0.2, 0.2, 0.35],
       width, color='cyan', edgecolor='k', label=r'$V$')
plt.xticks([1.15, 2.15, 3.15],
           [r'$S=1$', r'$S=2$', r'$S=3$'])
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.ylabel("Population Proportion")
plt.legend()
plt.tight_layout()
plt.savefig("figure_didactic_popprops.png", format='png', dpi=300)
plt.close()

# Calculating the true value
pop1 = generate_x(generate_data_1(n=1000000))
truth = np.mean(pop1['X'])
print(truth)
# print(np.mean(pop1['X*']))

d = generate_data(n1=1500, n2=750, n3=200)
d.to_csv("exdat.csv")
d.info()
# print(d.describe())
# sensitivity = np.mean(d.loc[(d['S'] == 3) & (d['X'] == 1), 'X*'])
# specificity = np.mean(d.loc[(d['S'] == 3) & (d['X'] == 0), 'X*'])
# print(sensitivity)
# print(specificity)
# print(np.mean(d.loc[d['S'] == 2, 'X*']))

s = np.asarray(d['S'])
x = np.asarray(d['X'])
xm = np.asarray(d['X*'])
d['C'] = 1
w = np.asarray(d[['C', 'W']])
p = np.where(d['S'] == 1, 1, np.nan)
p = np.where(d['S'] == 2, 0, p)


def psi_approach_2(theta):
    data = d.loc[d['S'] == 2].copy()
    return data['X*'] - theta


def psi_approach_3(theta):
    data = d.loc[d['S'] == 3].copy()
    return data['X'] - theta


def psi_approach_4(theta):
    sens = np.where(s == 3, (x == 1)*(xm-theta[0]), 0)
    spec = np.where(s == 3, (x == 0)*((1-xm)-theta[1]), 0)

    s_model = ee_logistic_regression(theta[4:],
                                     X=w, y=p)
    s_model = np.nan_to_num(s_model, copy=False, nan=0.)
    pi_s = inverse_logit(np.dot(w, theta[4:]))
    weight = pi_s / (1 - pi_s)

    mu_2 = np.where(s == 2, (xm - theta[2])*weight, 0)
    mu_1 = np.ones(s.shape)*theta[3]*(theta[0] + theta[1] - 1) - (theta[2] + theta[1] - 1)

    return np.vstack([sens[None, :],
                      spec[None, :],
                      mu_2[None, :],
                      mu_1[None, :],
                      s_model])


estr1 = MEstimator(psi_approach_2, init=[0.5])
estr1.estimate(solver='lm')

estr2 = MEstimator(psi_approach_3, init=[0.5])
estr2.estimate(solver='lm')

estr3 = MEstimator(psi_approach_4, init=[0.5, 0.5, 0.5, 0.5, 0., 0.])
estr3.estimate(solver='lm')

# Creating plot of results from a single data set
plt.vlines(truth, 0.5, 4.5, colors='gray', linestyles='--')

plt.scatter(estr1.theta[0], 3, s=100, color='blue')
plt.hlines(3, estr1.confidence_intervals()[0][0], estr1.confidence_intervals()[0][1], colors='blue')
plt.scatter(estr2.theta[0], 2, s=100, color='green')
plt.hlines(2, estr2.confidence_intervals()[0][0], estr2.confidence_intervals()[0][1], colors='green')
plt.scatter(estr3.theta[3], 1, s=100, color='k')
plt.hlines(1, estr3.confidence_intervals()[3][0], estr3.confidence_intervals()[3][1], colors='k')

plt.xlim([0.25, 0.75])
plt.xlabel(r"$\hat{\mu}$")
plt.ylim([0.5, 4.5])
plt.yticks([1, 2, 3, 4], ["Approach 4", "Approach 3", "Approach 2", "Approach 1"])
plt.tight_layout()
plt.savefig("didactic_example.png", format='png', dpi=300)
plt.close()

reps = 2000

bias_a2, cover_a2 = [], []
bias_a3, cover_a3 = [], []
bias_a4, cover_a4 = [], []

for i in range(reps):
    d = generate_data(n1=1500, n2=750, n3=200)

    s = np.asarray(d['S'])
    x = np.asarray(d['X'])
    xm = np.asarray(d['X*'])
    d['C'] = 1
    w = np.asarray(d[['C', 'W']])
    p = np.where(d['S'] == 1, 1, np.nan)
    p = np.where(d['S'] == 2, 0, p)

    # Approach 1
    # no analysis

    # Approach 2
    mest = MEstimator(psi_approach_2, init=[0.5])
    mest.estimate(solver='lm')
    est = mest.theta[0]
    ci = mest.confidence_intervals()[0]
    bias_a2.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_a2.append(1)
    else:
        cover_a2.append(0)

    # Approach 3
    mest = MEstimator(psi_approach_3, init=[0.5])
    mest.estimate(solver='lm')
    est = mest.theta[0]
    ci = mest.confidence_intervals()[0]
    bias_a3.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_a3.append(1)
    else:
        cover_a3.append(0)

    # Approach 4
    mest = MEstimator(psi_approach_4, init=[0.5, 0.5, 0.5, 0.5, 0., 0.])
    mest.estimate(solver='lm')
    est = mest.theta[3]
    ci = mest.confidence_intervals()[3]
    bias_a4.append(est - truth)
    if ci[0] < truth < ci[1]:
        cover_a4.append(1)
    else:
        cover_a4.append(0)


print(np.mean(bias_a2))
print(np.mean(cover_a2))

print(np.mean(bias_a3))
print(np.mean(cover_a3))

print(np.mean(bias_a4))
print(np.mean(cover_a4))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
# Reference line
ax.hlines([0], [0], [5], colors='gray', linestyles='--', zorder=1)

# Drawing violin plot results
parts = ax.violinplot([bias_a2, bias_a3, bias_a4], positions=[2, 3, 4],
                      showmeans=True, widths=0.35)
parts['bodies'][0].set_zorder(2)
for pc in parts['bodies']:
    pc.set_color("blue")
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
    vp = parts[partname]
    vp.set_edgecolor('blue')

ax2 = ax.twinx()
ax2.hlines([0.95], [0], [5], colors='gray', linestyles=':', zorder=1)
ax2.plot([2, 3, 4],
         [np.mean(cover_a2), np.mean(cover_a3), np.mean(cover_a4)],
         'D', color='mediumblue', markersize=7, zorder=3)

plt.xticks([1, 2, 3, 4],
           ["Approach 1", "Approach 2", "Approach 3", "Approach 4"])
plt.xlim([0.5, 4.5])
ax.set_ylim([-0.25, 0.25])
ax.set_ylabel(r"$\hat{\mu} - \mu$")
ax2.set_ylim([0, 1])
ax2.set_ylabel("95% CI Coverage")
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figure_didactic_results.png", format='png', dpi=300)
plt.close()
