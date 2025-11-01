import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Monte Carlo Portfolio Risk Simulator", layout="wide")

st.title("Monte Carlo Portfolio Risk Simulator")
st.markdown("""
This app estimates portfolio return distributions using Monte Carlo simulations.
Adjust the inputs on the left and click **Run Simulation** to explore risk outcomes.
""")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", 1000, 100000, 10000, step=1000)
time_horizon = st.sidebar.slider("Time Horizon (Years)", 1, 10, 1)

assets = st.sidebar.multiselect("Select Assets", ["AAPL", "MSFT", "GOOGL", "AMZN"], default=["AAPL", "MSFT", "GOOGL"])
num_assets = len(assets)

st.sidebar.subheader("Expected Returns & Volatilities")
mean_returns = []
volatilities = []
weights = []

for asset in assets:
    mean = st.sidebar.number_input(f"{asset} Expected Return (%)", value=8.0, key=f"mean_{asset}") / 100
    vol = st.sidebar.number_input(f"{asset} Volatility (%)", value=20.0, key=f"vol_{asset}") / 100
    weight = st.sidebar.slider(f"{asset} Weight", 0.0, 1.0, 1.0/num_assets, step=0.05)
    mean_returns.append(mean)
    volatilities.append(vol)
    weights.append(weight)

weights = np.array(weights) / np.sum(weights)  # normalize

# Simulation function
def simulate_portfolio(num_simulations, time_horizon, mean_returns, volatilities, weights):
    np.random.seed(42)
    dt = 1 / 252
    n_steps = int(252 * time_horizon)
    corr_matrix = np.eye(len(mean_returns))
    chol = np.linalg.cholesky(corr_matrix)
    portfolio_returns = []

    for _ in range(num_simulations):
        z = np.random.normal(size=(n_steps, len(mean_returns)))
        correlated_z = z @ chol.T
        daily_returns = np.exp((np.array(mean_returns) - 0.5 * np.array(volatilities)**2) * dt +
                               np.array(volatilities) * np.sqrt(dt) * correlated_z)
        portfolio_growth = np.prod(np.dot(daily_returns, weights))
        portfolio_returns.append(portfolio_growth - 1)

    return np.array(portfolio_returns)

# Run simulation
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulations..."):
        results = simulate_portfolio(num_simulations, time_horizon, mean_returns, volatilities, weights)
        mean_ret = np.mean(results)
        var_95 = np.percentile(results, 5)
        cvar_95 = results[results <= var_95].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return", f"{mean_ret*100:.2f}%")
        col2.metric("VaR (95%)", f"{var_95*100:.2f}%")
        col3.metric("CVaR (95%)", f"{cvar_95*100:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(results, bins=50, color="steelblue", alpha=0.7)
        ax.axvline(var_95, color="red", linestyle="--", label="VaR (95%)")
        ax.set_title("Distribution of Portfolio Returns")
        ax.set_xlabel("Simulated Portfolio Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)