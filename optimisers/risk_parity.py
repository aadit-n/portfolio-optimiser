import numpy as np
import pandas as pd
from scipy.optimize import minimize

def risk_parity_objective(w, cov):
    port_var = w.T @ cov @ w
    marginal_risk = cov @ w
    risk_contrib = w * marginal_risk
    risk_contrib_norm = risk_contrib / port_var
    return np.sum((risk_contrib_norm - 1.0 / len(w))**2)

def optimiser(data, cov_matrix):
    tickers = list(data.keys())
    n = len(tickers)
    cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    
    w0 = np.ones(n) / n  
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(risk_parity_objective, w0, args=(cov,), bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise ValueError("Risk parity optimization failed: " + result.message)
    
    return pd.Series(result.x, index=tickers)
