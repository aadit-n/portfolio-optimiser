import numpy as np
import pandas as pd

def optimiser(data, cov, n=10000, rfr=0.04):
    tickers = list(data.keys())
    num_assets = len(tickers)
    
    best_sharpe = -np.inf
    best_weights = None

    avg_return = np.array([data[t]['returns'].mean() * 252 for t in tickers])

    for _ in range(n):
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)

        port_return = avg_return @ weights
        port_vol = np.sqrt(weights.T @ cov @ weights)

        sharpe = (port_return - rfr) / port_vol

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    return pd.Series(best_weights, index=tickers)
