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
    view_type = st.sidebar.selectbox("View Type", ["Relative", "Absolute"])
    view_long = st.sidebar.text_input("Long Asset (for Relative)", "AAPL", key="bl_view_long")
    view_short = st.sidebar.text_input("Short Asset (for Relative)", "MSFT", key="bl_view_short")
    view_ticker = st.sidebar.text_input("Ticker (for Absolute)", "GOOG", key="bl_view_ticker")
    view_value = st.sidebar.number_input("Expected Return or Difference", value=0.05)

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
        result_df["Black-Litterman"] = bl_wts
        expected_return_df["Black-Litterman"] = np.dot(bl_wts, avg_return)
        volatility_df["Black-Litterman"] = np.sqrt(np.dot(bl_wts.T, np.dot(cov_matrix, bl_wts)))
        sharpe_ratio_df["Black-Litterman"] = (expected_return_df["Black-Litterman"] - risk_free_rate) / volatility_df["Black-Litterman"]  

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