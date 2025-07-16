import numpy as np
import pandas as pd


def empirical_covariance(data):
    returns = pd.DataFrame({ticker: data[ticker]['returns'] for ticker in data})
    returns = returns.dropna()
    return returns.cov()*252
