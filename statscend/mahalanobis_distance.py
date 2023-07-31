import numpy as np
import pandas as pd
from scipy.stats import chi2


def mahalanobis_distance(df=None, variables=None):

    x = np.array(df[variables])
    X_minus_mu = x - np.mean(x, axis=0)
    cov = np.cov(x.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(X_minus_mu, inv_covmat)
    mahal = np.dot(left_term, X_minus_mu.T).diagonal()

    # Calculate p-value for each Mahalanobis distance
    p_values = 1 - chi2.cdf(mahal, len(variables) - 1)

    # Add Mahalanobis distance and p_values to the DataFrame
    df['Mahalanobis'] = mahal
    df['p_values'] = p_values

    return df
