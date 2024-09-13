import numpy as np
import pandas as pd
from scipy.stats import logistic


def generate_data_ex1(n1, n2, missing=True):
    # Data for S=1
    d1 = pd.DataFrame()
    d1['S'] = [1, ]*n1

    # Data for S=2
    d2 = pd.DataFrame()
    d2['S'] = [2, ]*n2

    # Stacking data together
    d = pd.concat([d1, d2], ignore_index=True)

    # Generating Y and Y* for all observations
    sensitivity = 0.80
    specificity = 0.95
    pr_x = logistic.cdf(-0.5 + np.random.normal(scale=0.1, size=d.shape[0]))
    d['Y'] = np.random.binomial(n=1, p=pr_x, size=d.shape[0])
    d['Y*'] = np.random.binomial(n=1, p=sensitivity*d['Y'] + (1-specificity)*(1-d['Y']), size=d.shape[0])

    # Applying logic to make data set look like story
    if missing:
        d['Y'] = np.where(d['S'] == 2, d['Y'], np.nan)
    return d


def generate_data_ex2(n1, n2, n3, missing=True):
    # Data for S=1
    d1 = pd.DataFrame()
    d1['W'] = np.random.binomial(n=1, p=0.75, size=n1)
    d1['V'] = np.random.binomial(n=1, p=0.2, size=n1)
    d1['S'] = 1

    # Data for S=2
    d2 = pd.DataFrame()
    d2['W'] = np.random.binomial(n=1, p=0.65, size=n2)
    d2['V'] = np.random.binomial(n=1, p=0.35, size=n2)
    d2['S'] = 2

    # Data for S=3
    d3 = pd.DataFrame()
    d3['W'] = np.random.binomial(n=1, p=0.25, size=n3)
    d3['V'] = np.random.binomial(n=1, p=0.2, size=n3)
    d3['S'] = 3

    # Stacking data together
    d = pd.concat([d1, d2, d3], ignore_index=True)

    # Generating Y and Y* for all observations
    sensitivity = 0.80
    specificity = 0.95
    pr_x = logistic.cdf(-0.5 + 2*d['W'] - 1*d['V'] - 2*d['W']*d['V']
                        + np.random.normal(scale=0.1, size=d.shape[0]))
    d['Y'] = np.random.binomial(n=1, p=pr_x, size=d.shape[0])
    d['Y*'] = np.random.binomial(n=1, p=sensitivity*d['Y'] + (1-specificity)*(1-d['Y']), size=d.shape[0])

    # Applying logic to make data set look like story
    if missing:
        d['Y'] = np.where(d['S'] == 2, d['Y'], np.nan)
        d['W'] = np.where(d['S'] == 2, np.nan, d['W'])
    return d
