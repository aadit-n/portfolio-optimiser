import pandas as pd
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA


def shrinkage(data):
    
    returns = pd.DataFrame({ticker: data[ticker]['returns'] for ticker in data})
    returns = returns.dropna()
    X = returns.values
    cov_matrix_shrinked, _ = ledoit_wolf(X)
    return cov_matrix_shrinked*252

