import numpy as np
import pandas as pd
from scipy.stats import logistic


def generate_data_1(n):
    d = pd.DataFrame()
    d['W'] = np.random.binomial(n=1, p=0.25, size=n)
    d['V'] = np.random.binomial(n=1, p=0.2, size=n)
    d['S'] = 1
    return d


def generate_data_2(n):
    d = pd.DataFrame()
    d['W'] = np.random.binomial(n=1, p=0.75, size=n)
    d['V'] = np.random.binomial(n=1, p=0.2, size=n)
    d['S'] = 2
    return d


def generate_data_3(n):
    d = pd.DataFrame()
    d['W'] = np.random.binomial(n=1, p=0.65, size=n)
    d['V'] = np.random.binomial(n=1, p=0.35, size=n)
    d['S'] = 3
    return d


def generate_x(data):
    # sensitivity = np.where(data['V'] == 1, 0.95, 0.90)
    # specificity = np.where(data['V'] == 1, 0.99, 0.95)
    sensitivity = 0.80
    specificity = 0.95
    pr_x = logistic.cdf(-0.5 + 2*data['W'] - 1*data['V']
                        - 2*data['W']*data['V']
                        + np.random.normal(scale=0.1, size=data.shape[0])
                        )
    data['X'] = np.random.binomial(n=1, p=pr_x, size=data.shape[0])
    data['X*'] = np.random.binomial(n=1, p=sensitivity*data['X'] + (1-specificity)*(1-data['X']), size=data.shape[0])
    return data


def generate_data(n1, n2, n3):
    # Creating individual data sets
    d1 = generate_data_1(n=n1)
    d2 = generate_x(generate_data_2(n=n2))
    d3 = generate_x(generate_data_3(n=n3))

    # Stacking data sets together
    d = pd.concat([d1, d2, d3], ignore_index=True)

    # Applying logic to make data set look like story
    d['X'] = np.where(d['S'] == 3, d['X'], np.nan)
    d['V'] = np.where(d['S'] == 1, np.nan, d['V'])
    d['W'] = np.where(d['S'] == 3, np.nan, d['W'])
    return d
