# portfolio-optimiser

DISCLAIMER: NONE OF THE DATA OR RESULTS PROVIDED BY THIS WEBSITE/APP IS FINANCIAL ADVICE. ALL FINANCIAL DECISIONS MADE BY THE USER ARE INDEPENDENT OF THE RESULTS SHOWN HERE.


Portfolio-Optimiser is an advanced Streamlit app for portfolio optimization using Mean-Variance, Monte Carlo, Risk Parity, and Black-Litterman models. It supports custom asset views, visualizes efficient frontiers, and computes key metrics like Sharpe ratio, volatility, and risk contributions.

Try it out here!! --> [https://optimiseportfolio.streamlit.app](https://optimiseportfolio.streamlit.app)

## Features ##
# Multiple Optimization Strategies:

1. Mean-Variance Optimization (Modern Portfolio Theory)

2. Monte Carlo Simulation for stochastic portfolio sampling

3. Risk Parity Allocation based on equal risk contribution

4. Black-Litterman Model for incorporating subjective investor views



   <img width="1453" height="792" alt="image" src="https://github.com/user-attachments/assets/7a20e4a3-d037-427a-9a5c-4f431f8b2284" />


   
# Risk Models Supported:

1. Empirical Covariance Matrix (sample-based)

2. Ledoit-Wolf Shrinkage Estimator for more stable covariance estimation



   <img width="342" height="199" alt="image" src="https://github.com/user-attachments/assets/2d88f37a-03a8-49e0-8503-045373e01766" />



# Flexible Configuration:

1. Specify any number of stock tickers

2. Assign unique start dates for each asset

3. Set initial capital and risk-free rate

4. Toggle Black-Litterman views with full control over assumptions



   <img width="356" height="762" alt="image" src="https://github.com/user-attachments/assets/60564b6e-da06-42c5-9cf9-7bb7967dcf3e" />



   <img width="336" height="317" alt="image" src="https://github.com/user-attachments/assets/1075e188-7f97-41b6-b26f-4cbb8c0e7a0d" />



   <img width="348" height="664" alt="image" src="https://github.com/user-attachments/assets/acc10ffd-87c3-4234-884f-fe9407404c25" />



# Advanced Visualization Tools:

1. Efficient Frontier with Monte Carlo Simulated Portfolios



   <img width="1360" height="632" alt="image" src="https://github.com/user-attachments/assets/508f32c3-275a-4e10-863e-85ec4ea71404" />



3. Correlation Heatmap between asset returns



   <img width="1239" height="899" alt="image" src="https://github.com/user-attachments/assets/ba1d0b73-e9e6-4e79-931b-aa1691ea22a1" />



5. Portfolio weight breakdown and investment allocation



   <img width="1365" height="727" alt="image" src="https://github.com/user-attachments/assets/632f12c5-a473-44a3-82ce-efef3f0c2d31" />



6. Risk contribution plot for Risk-Parity portfolios



   <img width="1020" height="529" alt="image" src="https://github.com/user-attachments/assets/946dbf02-6a52-4138-addb-477d3a1e338f" />



# Key Portfolio Metrics:

1. Expected Annual Return

2. Portfolio Volatility

3. Sharpe Ratio

4. Risk Contribution per Asset



<img width="1484" height="883" alt="image" src="https://github.com/user-attachments/assets/bb3a3378-df5e-40b0-ae32-a6afb8c9d48b" />



## Installation ##
1. Clone the Repository

```bash
git clone https://github.com/aadit-n/portfolio-optimiser.git
cd portfolio-optimiser
```

2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

Run the Application

```bash
streamlit run app.py
```
