######################################################################################################
# M-estimation for Fusion Designs
#       Example 3: ACTG 320-175 Bridged Treatment Comparison
#
# Paul Zivich (2022/6/15)
######################################################################################################

# A copy of the data set used is available at:
#   https://github.com/pzivich/publications-code/tree/master/BridgeComparisonIntro
#
# Here, we use a different fitting strategy. Essentially, we will 'pre-wash' the covariates by finding
#   their solutions in pieces. This will make the optimization of the risk functions easier to do later
#   on. First, we will estimate the nuisance model parameters for each nuisance models. This 'pre-washing'
#   step will speed up our later calculations. After initializing those values, we will apply the M-estimator
#   for each of the unique event times. While we could theoretically solve all the risk functions simultaneously,
#   this can be difficult in practice due to run in practice. Therefore, we evaluate each unique event time, stack
#   together into lists, and then plot.

import patsy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression, ee_aft_weibull
from delicatessen.utilities import inverse_logit

#########################
# Pre-washing covariates


def psi_censor(theta):
    # Estimating Weibull AFT for censoring probabilities
    return ee_aft_weibull(theta=theta,
                          X=Xc,
                          delta=censor,
                          t=t)


def psi_sample(theta):
    # Selection model for inverse odds weights
    return ee_regression(theta=theta,
                         X=Xs,
                         y=s,
                         model='logistic')


def psi_treat(theta):
    # Treatment probabilities (marginal estimates)
    ee_mu_a2s1 = s*((a == 2) - theta[0])                       # Pr(A=2 | S=1)
    ee_mu_a1s1 = s*((a == 1) - theta[1])                       # Pr(A=1 | S=1)
    ee_mu_a1s0 = (1-s)*((a == 1) - theta[2])                   # Pr(A=1 | S=0)
    ee_mu_a0s0 = (1-s)*((a == 0) - theta[3])                   # Pr(A=0 | S=0)
    return np.vstack((ee_mu_a2s1, ee_mu_a1s1,
                      ee_mu_a1s0, ee_mu_a0s0))


# Reading in data
d = pd.read_csv("actg_data_formatted.csv", sep=",")      # See notes above for data availability
d = d.loc[(d['cd4'] >= 50) & (d['cd4'] <= 300)].copy()   # Restricting by CD4 counts

# Setting up variables as NumPy arrays
s = np.asarray(d['study'])
Xs = patsy.dmatrix("male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat)", d)
Xc = patsy.dmatrix("C(art) + study + male + black + idu + age + age_rs0 + age_rs1 + age_rs2 + C(karnof_cat)", d)[:, 1:]
a = np.asarray(d['art'])
delta = np.asarray(d['delta'])
censor = np.asarray(d['censor'])
t = np.asarray(d['t'])

# M-estimator for pre-washing all covariates
init_vals = [6.69, 0.1, 0.2, 0., 0., -1.2, 0., 0., -0.09, 0., 0., 0., 0., 1.12]
estr_c = MEstimator(psi_censor, init=init_vals)
estr_c.estimate(solver='hybr', maxiter=10000, tolerance=1e-9)

estr_s = MEstimator(psi_sample, init=[0., ]*10)
estr_s.estimate(solver='hybr', tolerance=1e-9)

estr_a = MEstimator(psi_treat, init=[0.5, 0.5, 0.5, 0.5])
estr_a.estimate(solver='hybr', tolerance=1e-9)

#########################
# Diagnostic


def pr_not_censored(time, X, mu, beta, gamma):
    """Quick function to calculate S(t) for a Weibull AFT model"""
    gamma = np.exp(gamma)
    lambd = np.exp(-1 * (mu + np.dot(X, beta)) * gamma)
    survival_t = np.exp(-1 * lambd * time**gamma)
    return survival_t


def psi_diagnostic(theta):
    """Estimating functions to estimate the nuisance models (prior to generating risk function plots)"""
    # Extracting coefficients for ease of reference
    rd, diag, r2, r11, r10, r0 = theta[:6]                                       # Risk Difference at selected times
    nuisance_parameters = theta[6:]                                              # Extract chunk of nuisance parameters
    param_aft = nuisance_parameters[0:14]                                        # Parameters for the Pr(C | A,W) model
    param_a2s1, param_a1s1, param_a1s0, param_a0s0 = nuisance_parameters[14:18]  # Parameters for the Pr(A | S) model
    param_logit = nuisance_parameters[18:]                                       # Parameters for the Pr(S | W) model

    # Selection model for inverse odds weights
    ee_logit = ee_regression(theta=param_logit,
                             X=Xs,
                             y=s,       # Logitistic model for the IOSW
                             model='logistic')
    pr_s1 = inverse_logit(np.dot(Xs, param_logit))             # Getting predicted probabilities
    odds = pr_s1 / (1-pr_s1)                                   # Converting to IOSW
    iosw = s*1 + (1-s)*odds                                    # Assigning IOSW

    # Treatment probabilities (marginal estimates)
    ee_mu_a2s1 = s*((a == 2) - param_a2s1)                       # Pr(A=2 | S=1)
    ee_mu_a1s1 = s*((a == 1) - param_a1s1)                       # Pr(A=1 | S=1)
    ee_mu_a1s0 = (1-s)*((a == 1) - param_a1s0)                   # Pr(A=1 | S=0)
    ee_mu_a0s0 = (1-s)*((a == 0) - param_a0s0)                   # Pr(A=0 | S=0)
    # Assigning IPTW based on pattern observed
    pr_a = (s*(a == 2)*param_a2s1 + s*(a == 1)*param_a1s1 + (1-s)*(a == 1)*param_a1s0 + (1-s)*(a == 0)*param_a0s0)

    # Estimating Weibull AFT for censoring probabilities
    ee_aft = ee_aft_weibull(theta=param_aft,
                            X=Xc,
                            delta=censor,
                            t=t)
    pr_c = pr_not_censored(time=t, X=Xc, mu=param_aft[0], beta=param_aft[1:-1], gamma=param_aft[-1])

    # Weighted EDF
    tdelta = (t <= t_index) * delta
    ee_r_a2_s1 = s * (((a == 2) * tdelta) / (pr_a * pr_c) - r2)
    ee_r_a1_s1 = s * (((a == 1) * tdelta) / (pr_a * pr_c) - r11)
    ee_r_a1_s0 = (1-s) * iosw * (((a == 1) * tdelta) / (pr_a * pr_c) - r10)
    ee_r_a0_s0 = (1-s) * iosw * (((a == 0) * tdelta) / (pr_a * pr_c) - r0)

    # Bridged diagnostic
    ee_diag = np.ones(t.shape[0])*(r11 - r10) - diag

    # Bridged parameter of interest
    ee_brdg = np.ones(t.shape[0])*((r2 - r11) + (r10 - r0)) - rd

    # Returning the stacked estimating function evaluations
    return np.vstack((ee_brdg, ee_diag,
                      ee_r_a2_s1, ee_r_a1_s1, ee_r_a1_s0, ee_r_a0_s0,
                      ee_aft,
                      ee_mu_a2s1, ee_mu_a1s1, ee_mu_a1s0, ee_mu_a0s0,
                      ee_logit))


# Generating info for plotting
# NOTE: this process could be sped up by running in parallel using Pool
event_t = [0, ] + sorted(np.unique(d.loc[d['delta'] == 1, 't'])) + [np.max(d['t'])]
brdg_est, brdg_lcl, brdg_ucl = [], [], []
diag_est, diag_lcl, diag_ucl = [], [], []
init_bridge = [0, 0, 0, 0, 0, 0, ]
for t_index in event_t:
    init_vals = (init_bridge
                 + list(estr_c.theta)
                 + list(estr_a.theta)
                 + list(estr_s.theta))
    estr = MEstimator(psi_diagnostic, init=init_vals)
    estr.estimate(solver='hybr', maxiter=10000, tolerance=1e-7)

    # Storing the output
    brdg_est.append(estr.theta[0])
    brdg_ci = estr.confidence_intervals()[0, :]
    brdg_lcl.append(brdg_ci[0])
    brdg_ucl.append(brdg_ci[1])

    diag_est.append(estr.theta[1])
    diag_ci = estr.confidence_intervals()[1, :]
    diag_lcl.append(diag_ci[0])
    diag_ucl.append(diag_ci[1])

    # Using previous inits (for general faster run-times)
    init_bridge = list(estr.theta[:6])


# Generating 1-by-2 plot of the results
f, ax = plt.subplots(1, 2, figsize=(6.75, 4.5))

# Diagnostic plot
ax[0].vlines(0, 0, 370, colors='gray', linestyles='--')
ax[0].step(diag_est, [event_t[1], ] + event_t[2:] + [event_t[-1]], color='purple', where='post')
ax[0].fill_betweenx(event_t, diag_lcl, diag_ucl, alpha=0.2, color='purple', step='post')
ax[0].set_xlabel("Difference in Shared")
ax[0].set_ylabel("Time (days)")
ax[0].set_xlim([-0.2, 0.2])
ax[0].set_ylim([0, 370])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax02 = ax[0].twiny()  # Duplicate the x-axis to create a separate label
ax02.set_xlabel("Favors 320" + "\t".expandtabs() + "Favors 175", fontdict={"size": 10})
ax02.set_xticks([])
ax02.xaxis.set_ticks_position('bottom')
ax02.xaxis.set_label_position('bottom')
ax02.spines['bottom'].set_position(('outward', 36))
ax02.spines['top'].set_visible(False)
ax02.spines['right'].set_visible(False)
ax[0].set_title("Diagnostic")

# Risk Difference of Interest plot
ax[1].vlines(0, 0, 370, colors='gray', linestyles='--')
ax[1].step(brdg_est, [event_t[1], ] + event_t[2:] + [event_t[-1]], color='mediumblue', where='post')
ax[1].fill_betweenx(event_t, brdg_lcl, brdg_ucl, alpha=0.2, color='mediumblue', step='post')
ax[1].set_xlabel("Risk Difference")
ax[1].set_xlim([-0.4, 0.4])
ax[1].set_ylim([0, 370])
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

ax12 = ax[1].twiny()  # Duplicate the x-axis to create a separate label
ax12.set_xlabel("Favors Triple" + "\t".expandtabs() + "Favors Mono  ", fontdict={"size": 10})
ax12.set_xticks([])
ax12.xaxis.set_ticks_position('bottom')
ax12.xaxis.set_label_position('bottom')
ax12.spines['bottom'].set_position(('outward', 36))
ax12.spines['top'].set_visible(False)
ax12.spines['right'].set_visible(False)
ax[1].set_title("Interest Parameter")

# Saving plot for slides
plt.tight_layout()
plt.savefig("../images/ex_a3_results.png", format='png', dpi=300)
plt.close()
