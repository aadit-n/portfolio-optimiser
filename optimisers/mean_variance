import pandas as pd
import numpy as np
import cvxpy as cp

def optimiser(data, cov, rfr=0.0428):
    tickers = list(data.keys())
    n = len(tickers)

    weights = cp.Variable(n)
    avg_return = np.array([data[t]['returns'].mean() * 252 for t in tickers])

    Sigma = cov.values if isinstance(cov, pd.DataFrame) else np.array(cov)
    Sigma = np.asarray(Sigma)

    port_return = avg_return @ weights

    port_var = cp.quad_form(weights, Sigma)   

    cp_prob = cp.Problem(cp.Minimize(port_var),
                         [cp.sum(weights) == 1, weights >= 0])
    cp_prob.solve()

    return pd.Series(weights.value, index=tickers)
