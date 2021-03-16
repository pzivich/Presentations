import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import logistic
from pygam import LinearGAM
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from zepid.causal.causalgraph import DirectedAcyclicGraph

np.random.seed(20210316)

###############################################
# Directed Acyclic Graph
###############################################
dag = DirectedAcyclicGraph(exposure=r'$X$', outcome=r'$Y$')
dag.add_arrows([[r'$Z$', r'$X$'],
                [r'$Z$', r'$Y$'],
                [r'$X$', r'$M$'],
                [r'$M$', r'$Y$'],
                [r'$U$', r'$Z$'],
                [r'$U$', r'$X$'],
                ])

dag.draw_dag(positions={r'$Z$': [-0.75, 0.05],
                        r'$X$': [-0.5, 0],
                        r'$M$': [-0.25, -0.02],
                        r'$U$': [-1, 0],
                        r'$Y$': [0, 0]},
             fig_size=[6, 3])
plt.ylim([-0.05, 0.08])
plt.savefig("../images/dag.png", format='png', dpi=300)
plt.close()

###############################################
# Convergence Plot
###############################################
x = np.linspace(0, 10, 200)
y1 = -0.5*x
y2 = -0.3*x - 0.35

plt.plot(x, y1, color="dimgray")
plt.plot([1.1, 4, 6, 9], [-0.5, -2.03, -2.98, -4.51], 'o', color='k')
plt.text(4.2, -2.7, "-0.5")
plt.plot(x, y2, "--", color='gray')
plt.plot([1.1, 4, 6, 9], [x-0.35 for x in [-0.29, -1.25, -1.81, -2.71]], 's', color='k')
plt.text(6, -2, "-0.3")

plt.ylim([-6.2, 0.3])
plt.yticks([])
plt.xlabel(r"$\log(n)$")
plt.xticks([])
plt.ylabel("Empirical Standard Error")

plt.tight_layout()
plt.savefig("../images/convergence.png", format='png', dpi=300)
plt.close()

###############################################
# Categorization Approach
###############################################
n = 200
nsims = 1000
plot = False

z = np.random.normal(size=100000)
yx1 = -1*1 + 0.8*1*(z > 0.5) + 0.9*z + 0.5*z*z + np.random.normal(size=100000)
yx0 = -1*0 + 0.8*0*(z > 0.5) + 0.9*z + 0.5*z*z + np.random.normal(size=100000)
truth = np.mean(yx1 - yx0)

# Simulation Loop
est_ace = []
filenames = []
for i in range(nsims):
    z = np.random.normal(size=n)
    x = np.random.binomial(n=1, p=logistic.cdf(-0.1 + 0.2*z), size=n)
    y = -1*x - 0.8*x*(z > 0.5) + 0.9*z + 0.5*z*z + np.random.normal(size=n)

    # non-parametric G-formula
    y_x1 = (np.mean(y[(x == 1) & (z <= -1)])*np.mean((z <= -1)) +
            np.mean(y[(x == 1) & (-1 < z) & (z < 1)])*np.mean(((-1 < z) & (z < 1))) +
            np.mean(y[(x == 1) & (z >= 1)])*np.mean((z >= 1)))
    y_x0 = (np.mean(y[(x == 0) & (z <= -1)])*np.mean((z <= -1)) +
            np.mean(y[(x == 0) & (-1 < z) & (z < 1)])*np.mean(((-1 < z) & (z < 1))) +
            np.mean(y[(x == 0) & (z >= 1)])*np.mean((z >= 1)))
    ace = y_x1 - y_x0
    est_ace.append(ace)

    if plot:
        plt.figure(figsize=[8, 5])
        plt.subplot(121)
        plt.plot(z[x == 0], y[x == 0], 's', color='r', markeredgecolor='k', zorder=0)
        plt.hlines(np.mean(y[(x == 0) & (z <= -1)]), -3.5, -1, colors='firebrick', linestyle='-', zorder=1)
        plt.hlines(np.mean(y[(x == 0) & (-1 < z) & (z < 1)]), -1, 1, colors='firebrick', linestyle='-', zorder=1)
        plt.hlines(np.mean(y[(x == 0) & (z >= 1)]), 1, 3.5, colors='firebrick', linestyle='-', zorder=1)

        plt.plot(z[x == 1], y[x == 1], 'o', color='b', markeredgecolor='k', zorder=0)
        plt.hlines(np.mean(y[(x == 1) & (z <= -1)]), -3.5, -1, colors='navy', linestyle='-', zorder=1)
        plt.hlines(np.mean(y[(x == 1) & (-1 < z) & (z < 1)]), -1, 1, colors='navy', linestyle='-', zorder=1)
        plt.hlines(np.mean(y[(x == 1) & (z >= 1)]), 1, 3.5, colors='navy', linestyle='-', zorder=1)

        plt.vlines([-1, 1], -6, 6, colors='k', linestyles=':')
        plt.ylim([-4, 6])
        plt.ylabel(r"$Y$")
        plt.xlim([-3.5, 3.5])
        plt.xlabel(r"$Z$")
        plt.title("Data set")

        plt.subplot(122)
        bins = np.arange(-2, -0.5, 0.03)
        plt.hist(est_ace, bins=bins, color='dimgray')
        plt.vlines(truth, 0, 100, colors='blue')
        plt.vlines(np.mean(est_ace), 0, 100, colors='red', linestyles='--')

        plt.ylim([0, 80])
        plt.xlabel(r"$E[Y(1)]-E[Y(0)]$")
        plt.yticks([])
        plt.title("Estimated ACE")
        plt.tight_layout()

        fname = f'{i}.png'
        filenames.append(fname)
        plt.savefig(fname)
        plt.close()


# build gif
if plot:
    with imageio.get_writer('../categorization.mov', fps=15) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

print("BIAS:", np.round(np.mean(est_ace), 3))

###############################################
# Simulation Results
###############################################
file_path = "/home/pzivich/Documents/Research/ABreskin/DoubleCrossfit/Python Code/results/data_files/"

files = ["aipw_true.csv",
         "tmle_true.csv",
         "dcaipw_true.csv",
         "dctmle_true.csv"]
spots = [0.8,
         1.8,
         2.8,
         3.8]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4])
ax.hlines([0], [0.5], [6.5], colors='gray', linestyles='--')
ax2 = ax.twinx()
ax2.hlines([0.95], [0.5], [6.5], colors='gray', linestyles=':')

for i, j in zip(files, spots):
    # Loading data results
    df = pd.read_csv(file_path+i).dropna()

    # Violinplot
    parts = ax.violinplot([df['bias']], positions=[j],
                          showmeans=True, showmedians=False, widths=0.25)
    for pc in parts['bodies']:
        if 'true' in i:
            pc.set_color('#1A8B71')
            pc.set_alpha(0.6)
        elif 'main' in i:
            pc.set_color('sandybrown')
            pc.set_alpha(0.6)
        elif 'ml' in i:
            pc.set_color('#0B4358')
            pc.set_alpha(0.6)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        if 'true' in i:
            vp.set_color('#1A8B71')
        elif 'main' in i:
            vp.set_color('sandybrown')
        elif 'ml' in i:
            vp.set_color('#0B4358')

    # Adding CL Coverage
    if 'true' in i:
        cl_color = '#1A8B71'
    elif 'main' in i:
        cl_color = 'sandybrown'
    elif 'ml' in i:
        cl_color = '#0B4358'

    ax2.plot(j, np.mean(df['cover']), 'o', color=cl_color)


ax.set_ylim([-0.11, 0.11])
ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
ax.set_ylabel("Bias")
ax2.set_ylim([0.0, 1])
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
ax2.set_ylabel("95 % CL Coverage")

ax.set_xlim([0.5, 4.5])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["AIPW", "TMLE", "DC-AIPW", "DC-TMLE"])

plt.tight_layout()
plt.savefig("../images/sim_result1.png", format='png', dpi=300)
plt.close()

files = ["aipw_true.csv", "aipw_main.csv",
         "tmle_true.csv", "tmle_main.csv",
         "dcaipw_true.csv", "dcaipw_main.csv",
         "dctmle_true.csv", "dctmle_main.csv"]
spots = [0.8, 1.0,
         1.8, 2.0,
         2.8, 3.0,
         3.8, 4.0]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4])
ax.hlines([0], [0.5], [6.5], colors='gray', linestyles='--')
ax2 = ax.twinx()
ax2.hlines([0.95], [0.5], [6.5], colors='gray', linestyles=':')

for i, j in zip(files, spots):
    # Loading data results
    df = pd.read_csv(file_path+i).dropna()

    # Violinplot
    parts = ax.violinplot([df['bias']], positions=[j],
                          showmeans=True, showmedians=False, widths=0.25)
    for pc in parts['bodies']:
        if 'true' in i:
            pc.set_color('#1A8B71')
            pc.set_alpha(0.6)
        elif 'main' in i:
            pc.set_color('sandybrown')
            pc.set_alpha(0.6)
        elif 'ml' in i:
            pc.set_color('#0B4358')
            pc.set_alpha(0.6)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        if 'true' in i:
            vp.set_color('#1A8B71')
        elif 'main' in i:
            vp.set_color('sandybrown')
        elif 'ml' in i:
            vp.set_color('#0B4358')

    # Adding CL Coverage
    if 'true' in i:
        cl_color = '#1A8B71'
    elif 'main' in i:
        cl_color = 'sandybrown'
    elif 'ml' in i:
        cl_color = '#0B4358'

    ax2.plot(j, np.mean(df['cover']), 'o', color=cl_color)


ax.set_ylim([-0.11, 0.11])
ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
ax.set_ylabel("Bias")
ax2.set_ylim([0.0, 1])
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
ax2.set_ylabel("95 % CL Coverage")

ax.set_xlim([0.5, 4.5])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["AIPW", "TMLE", "DC-AIPW", "DC-TMLE"])

plt.tight_layout()
plt.savefig("../images/sim_result2.png", format='png', dpi=300)
plt.close()

files = ["aipw_true.csv", "aipw_main.csv", "aipw_ml.csv",
         "tmle_true.csv", "tmle_main.csv", "tmle_ml.csv",
         "dcaipw_true.csv", "dcaipw_main.csv", "dcaipw_ml.csv",
         "dctmle_true.csv", "dctmle_main.csv", "dctmle_ml.csv"]
spots = [0.8, 1.0, 1.2,
         1.8, 2.0, 2.2,
         2.8, 3.0, 3.2,
         3.8, 4.0, 4.2]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 4])
ax.hlines([0], [0.5], [6.5], colors='gray', linestyles='--')
ax2 = ax.twinx()
ax2.hlines([0.95], [0.5], [6.5], colors='gray', linestyles=':')

for i, j in zip(files, spots):
    # Loading data results
    df = pd.read_csv(file_path+i).dropna()

    # Violinplot
    parts = ax.violinplot([df['bias']], positions=[j],
                          showmeans=True, showmedians=False, widths=0.25)
    for pc in parts['bodies']:
        if 'true' in i:
            pc.set_color('#1A8B71')
            pc.set_alpha(0.6)
        elif 'main' in i:
            pc.set_color('sandybrown')
            pc.set_alpha(0.6)
        elif 'ml' in i:
            pc.set_color('#0B4358')
            pc.set_alpha(0.6)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        if 'true' in i:
            vp.set_color('#1A8B71')
        elif 'main' in i:
            vp.set_color('sandybrown')
        elif 'ml' in i:
            vp.set_color('#0B4358')

    # Adding CL Coverage
    if 'true' in i:
        cl_color = '#1A8B71'
    elif 'main' in i:
        cl_color = 'sandybrown'
    elif 'ml' in i:
        cl_color = '#0B4358'

    ax2.plot(j, np.mean(df['cover']), 'o', color=cl_color)


ax.set_ylim([-0.11, 0.11])
ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
ax.set_ylabel("Bias")
ax2.set_ylim([0.0, 1])
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
ax2.set_ylabel("95 % CL Coverage")

ax.set_xlim([0.5, 4.5])
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["AIPW", "TMLE", "DC-AIPW", "DC-TMLE"])

plt.tight_layout()
plt.savefig("../images/sim_result3.png", format='png', dpi=300)
plt.close()
