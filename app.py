import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import cvxpy as cp
from fetch_data import fetch_data
from optimisers.mean_variance import optimiser as mean_variance_opt
from optimisers.monte_carlo import optimiser as  monte_carlo_optimize
from optimisers.risk_parity import optimiser as risk_parity_optimizer
from optimisers.black_litterman import optimiser as black_litterman_opt
from risk_models.empirical_covariance import empirical_covariance
from risk_models.ledoit_wolf_shrinkage import shrinkage

st.set_page_config(layout="wide")
st.title("Advanced Portfolio Optimizer")
st.markdown("""
DISCLAIMER: NONE OF THE DATA OR RESULTS PROVIDED BY THIS WEBSITE/APP IS FINANCIAL ADVICE. ALL FINANCIAL DECISIONS MADE BY THE USER ARE INDEPENDENT OF THE RESULTS SHOWN HERE.
""")
with st.expander("How This App Works (Math & Concepts)"):
    st.header("Portfolio Optimization Methods")

    st.subheader("1. Mean-Variance Optimization")
    st.markdown("""
    This is based on **Modern Portfolio Theory** by Harry Markowitz. The goal is to find the optimal weights that either:
    
    - Minimize risk (portfolio variance) for a given return, or  
    - Maximize return for a given level of risk.

    The mathematical formulation is:
    """)
    st.latex(r"""
    \text{Minimize } \quad w^T \Sigma w \quad \text{subject to} \quad \sum w_i = 1, \quad w_i \geq 0
    """)
    st.markdown("""
    Where:
    - \( w \) = asset weights  
    - \(Σ) = covariance matrix of asset returns  
    - Constraints: weights sum to 1 and are non-negative (long-only)
    """)

    st.subheader("2. Monte Carlo Simulation")
    st.markdown("""
    This method randomly generates thousands of valid portfolios with different weight combinations. For each, we calculate:
    - **Expected Return**
    - **Volatility**
    - **Sharpe Ratio**

    Then we select the one with the **highest Sharpe Ratio**.
    """)
    st.latex(r"""
    \text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
    """)
    st.markdown("""
    Where:
    - \( R_p \) = expected portfolio return  
    - \( R_f \) = risk-free rate (often assumed 0 for simplification)  
    - \( \sigma_p \) = standard deviation of portfolio returns
    """)

    st.subheader("3. Risk Parity")
    st.markdown("""
    Risk parity allocates weights so that **each asset contributes equally to total portfolio risk**.

    Risk contribution of asset \( i \) is:
    """)
    st.latex(r"""
    RC_i = w_i (\Sigma w)_i
    """)
    st.markdown("""
    Objective:
    - Equalize all \( RC_i \)
    - Usually solved by minimizing squared deviations from average risk contribution.

    This method avoids concentration in low-volatility assets and achieves better diversification of **risk**, not just capital.
    """)

    st.subheader("4. Black-Litterman Model")
    st.markdown("""
    This model blends:
    - Market-implied equilibrium returns  
    - Subjective views from investors  

    It overcomes weaknesses in standard mean-variance optimization by allowing for more stable and realistic return estimates.
    """)

    st.latex(r"""
    \mu_{BL} = \pi + \tau \Sigma P^T (P \tau \Sigma P^T + \Omega)^{-1} (Q - P \pi)
    """)

    st.markdown("""
    Where:
    - \(π) = implied equilibrium excess returns  
    - \(τ) = scaling factor for uncertainty in \( \pi \)  
    - \(Σ) = covariance matrix  
    - \( P \) = matrix that encodes investor views  
    - \( Q \) = expected returns from views  
    - \(Ω) = uncertainty (covariance) in views
    """)

    st.subheader("5. Risk Models")
    st.markdown("""
    You can choose between two types of covariance estimators:

    - **Empirical Covariance**:
        - Standard sample covariance matrix calculated from historical returns.
        - Can be unstable when number of assets is large vs number of observations.

    - **Ledoit-Wolf Shrinkage**:
        - Improves stability and invertibility by combining the sample covariance with a structured matrix (like identity).
        - Especially useful when using advanced optimizers like risk parity or Black-Litterman.

    The formula is:
    """)

    st.latex(r"""
    \Sigma_{LW} = (1 - \alpha) \Sigma_{\text{sample}} + \alpha F
    """)
    st.markdown("""
    Where:
    - \( Σ_{\text{sample}} \) = empirical covariance  
    - \( F \) = structured target (e.g., identity or constant correlation)  
    - \( α \in [0, 1] \) = shrinkage intensity
    """)


st.sidebar.header("User Inputs")

ticker_input = st.sidebar.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOG,AMZN,TSLA")
tickers = [t.strip().upper() for t in ticker_input.split(",")]

st.sidebar.markdown("### Start Dates per Ticker")
start_dates = {}
for ticker in tickers:
    start_dates[ticker] = st.sidebar.date_input(f"Start Date for {ticker}", pd.to_datetime("2020-01-01"), key=f"start_{ticker}")


capital = st.sidebar.number_input("Portfolio Capital ($)", value=10000.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=0.04)

risk_model = st.sidebar.selectbox("Select Risk Model", ["Empirical Covariance", "Ledoit-Wolf Shrinkage"])

use_black_litterman = st.sidebar.checkbox("Use Black-Litterman")

if use_black_litterman:
    st.sidebar.subheader("Black-Litterman Views")
    st.sidebar.markdown("""
    The Black-Litterman model blends market equilibrium returns with your subjective views.  
    - Select whether your view is **Relative** (e.g., AAPL will outperform MSFT)  
    - or **Absolute** (e.g., GOOG will return 7% annually).
    """)

    view_type = st.sidebar.selectbox("View Type", ["Relative", "Absolute"])
    st.sidebar.markdown("""
    **Relative:** Compare two assets (e.g., AAPL - MSFT > 5%)  
    **Absolute:** Set expected return for one asset (e.g., GOOG = 7%)
    """)

    if view_type == "Relative":
        view_long = st.sidebar.text_input("Long Asset (expected to outperform)", "AAPL", key="bl_view_long")
        st.sidebar.caption("This is the asset you believe will perform **better**.")
        view_short = st.sidebar.text_input("Short Asset (expected to underperform)", "MSFT", key="bl_view_short")
        st.sidebar.caption("This is the asset you believe will perform **worse**.")
    else:
        view_ticker = st.sidebar.text_input("Ticker for Absolute View", "GOOG", key="bl_view_ticker")
        st.sidebar.caption("Enter the asset for which you have a specific return expectation.")

    view_value = st.sidebar.number_input("Expected Return or Difference", value=0.05)
    st.sidebar.caption("""
    - For **relative** views: Enter the expected **difference in returns** (e.g., 0.05 = 5% more return than the other asset)  
    - For **absolute** views: Enter the expected **annual return** (e.g., 0.07 = 7%)
    """)


if st.button("Run Optimization"):
    st.write(f"Fetching data for: {', '.join(tickers)}")
    data = {}
    for ticker in tickers:
        data[ticker] = fetch_data([ticker], start_dates[ticker])[ticker]

    if risk_model == "Empirical Covariance":
        cov_matrix = empirical_covariance(data)

    else:
        cov_matrix = shrinkage(data)
        cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)


    st.subheader("Optimized Portfolio Weights")
    avg_return = np.array([data[t]['returns'].mean() * 252 for t in tickers])
    mean_var_wts = mean_variance_opt(data, cov_matrix, rfr=risk_free_rate)
    mc_wts = monte_carlo_optimize(data, cov_matrix, rfr=risk_free_rate)
    risk_parity_wts = risk_parity_optimizer(data, cov_matrix)

    result_df = pd.DataFrame({
        "Mean-Variance": mean_var_wts,
        "Monte Carlo": mc_wts,
        "Risk Parity": risk_parity_wts
    })

    expected_return_df = pd.DataFrame({
        "Mean-Variance": [np.dot(mean_var_wts, avg_return)],
        "Monte Carlo": [np.dot(mc_wts, avg_return)],
        "Risk Parity": [np.dot(risk_parity_wts, avg_return)]
    }, index=["Expected Return"])

    volatility_df = pd.DataFrame({
        "Mean-Variance": [np.sqrt(mean_var_wts @ cov_matrix @ mean_var_wts)],
        "Monte Carlo": [np.sqrt(mc_wts @ cov_matrix @ mc_wts)],
        "Risk Parity": [np.sqrt(risk_parity_wts @ cov_matrix @ risk_parity_wts)]
    }, index=["Volatility"])

    sharpe_ratio_df = pd.DataFrame({
        "Mean-Variance": [(expected_return_df["Mean-Variance"][0] - risk_free_rate) / volatility_df["Mean-Variance"][0]],
        "Monte Carlo": [(expected_return_df["Monte Carlo"][0] - risk_free_rate) / volatility_df["Monte Carlo"][0]],
        "Risk Parity": [(expected_return_df["Risk Parity"][0] - risk_free_rate) / volatility_df["Risk Parity"][0]]
    }, index=["Sharpe Ratio"])


    if use_black_litterman:
        market_weights = pd.Series([1 / len(tickers)] * len(tickers), index=tickers)
        if view_type == "Relative":
            P = np.zeros((1, len(tickers)))
            P[0][tickers.index(view_long)] = 1
            P[0][tickers.index(view_short)] = -1
            Q = np.array([view_value])
        else:
            P = np.zeros((1, len(tickers)))
            P[0][tickers.index(view_ticker)] = 1
            Q = np.array([view_value])

        bl_wts = black_litterman_opt(cov_matrix, market_weights, P, Q)

        if bl_wts is not None and not bl_wts.isnull().values.any():
            result_df["Black-Litterman"] = bl_wts
            expected_return_df["Black-Litterman"] = [np.dot(bl_wts, avg_return)]
            volatility_df["Black-Litterman"] = [np.sqrt(bl_wts.T @ cov_matrix @ bl_wts)]
            bl_exp_ret = expected_return_df["Black-Litterman"].iloc[0]
            bl_vol = volatility_df["Black-Litterman"][0]
            sharpe_ratio_df["Black-Litterman"] = [(bl_exp_ret - risk_free_rate) / bl_vol]
        else:
            st.warning("Black-Litterman optimization failed. Please check your views or try different parameters.")

    marginal_contrib = cov_matrix @ risk_parity_wts
    risk_contrib =   risk_parity_wts*marginal_contrib
    risk_contrib_pct = risk_contrib/np.dot(risk_parity_wts.T, marginal_contrib)

    st.dataframe(result_df.style.format("{:.2%}"))

    st.subheader("Investment Allocation (in $)")
    investment_df = result_df.multiply(capital, axis=0)
    st.dataframe(investment_df.style.format("${:,.2f}"))

    st.subheader("Metrics")
    st.text("Expected Return")
    st.dataframe(expected_return_df.style.format("{:.2%}"))
    st.text("Volatility")
    st.dataframe(volatility_df.style.format("{:.2%}"))
    st.text("Sharpe Ratio")
    st.dataframe(sharpe_ratio_df.style.format("{:.2f}"))

    st.subheader("Risk Contribution per Ticker from Risk-Parity Optimisation")
    st.bar_chart(pd.Series(risk_contrib_pct, index=tickers))

    returns_df = pd.DataFrame({t: data[t]['returns'] for t in data}).dropna()
    correlation_matrix = returns_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.subheader("Correlation Matrix")
    st.pyplot(plt.gcf())

    returns_list = []
    vols_list = []
    sharpe_list = []

    for _ in range(5000):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        r = np.dot(w, avg_return)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        s = (r - risk_free_rate) / v

        returns_list.append(r)
        vols_list.append(v)
        sharpe_list.append(s)

    df = pd.DataFrame({'Return': returns_list, 'Volatility': vols_list, 'Sharpe': sharpe_list})

    min_ret, max_ret = np.percentile(returns_list, [1, 99])
    target_returns = np.linspace(min_ret, max_ret, 100)
    frontier_returns = []
    frontier_vols = []

    n = len(tickers)
    w = cp.Variable(n)

    for target in target_returns:
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            avg_return @ w == target
        ]
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status == "optimal" and w.value is not None:
            frontier_returns.append(target)
            frontier_vols.append(np.sqrt(w.value.T @ cov_matrix @ w.value))
    max_sharpe_idx = np.argmax(df['Sharpe'])

    fig = px.scatter(df, x='Volatility', y='Return', color='Sharpe',
                    title='Efficient Frontier with Monte Carlo Portfolios',
                    opacity=0.5)

    fig.add_scatter(x=[df.loc[max_sharpe_idx, 'Volatility']],
                y=[df.loc[max_sharpe_idx, 'Return']],
                mode='markers',
                name='Max Sharpe Portfolio',
                marker=dict(color='green', size=10, symbol='star'))

    st.subheader("Efficient Frontier")
    st.plotly_chart(fig)

