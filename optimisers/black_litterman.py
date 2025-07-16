import numpy as np
import pandas as pd
import cvxpy as cp

def optimiser(cov_matrix, market_weights, views_P, views_Q, tau=0.05, delta=2.5):

    tickers = market_weights.index if isinstance(market_weights, pd.Series) else [f"Asset {i}" for i in range(len(market_weights))]
    n = len(market_weights)

    w_mkt = np.array(market_weights)
    Sigma = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
    P = np.array(views_P)
    Q = np.array(views_Q)

    pi = delta * Sigma @ w_mkt

    tau_sigma = tau * Sigma
    omega = np.diag(np.diag(P @ tau_sigma @ P.T))  

    middle_term = np.linalg.inv(P @ tau_sigma @ P.T + omega)
    mu_bl = pi + tau_sigma @ P.T @ middle_term @ (Q - P @ pi)

    w = cp.Variable(n)
    port_return = mu_bl @ w
    port_var = cp.quad_form(w, Sigma)
    objective = cp.Maximize(port_return - 0.5 * delta * port_var)

    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return pd.Series(w.value, index=tickers)
