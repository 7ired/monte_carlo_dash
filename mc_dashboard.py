import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yfinance as yf
from datetime import datetime as dt
import seaborn as sns
from warnings import simplefilter
simplefilter('ignore')
yf.pdr_override()

# Function to perform simulation
def monte_carlo_simulation(stock, T, sims, confidence_level):
    df = yf.download(stock, start_date, end_date)[['Close']]
    daily_returns = df['Close'].pct_change().dropna()
    daily_mean = np.mean(daily_returns)
    daily_volatility = np.std(daily_returns)

    starting_price = df['Close'][-1]

    returns_matrix = np.zeros(shape=(T, sims))
    prices_matrix = np.zeros(shape=(T, sims))

    for i in range(sims):
        returns_matrix[:,i] = np.random.normal(size=T, loc=daily_mean, scale=daily_volatility)
        prices_matrix[:,i] = np.cumprod(1 + returns_matrix[:,i]) * starting_price

    var = np.percentile(prices_matrix[-1,:], confidence_level * 100)
    cvar = np.mean(prices_matrix[-1,prices_matrix[-1,:] <= var])

    return var, cvar, prices_matrix

# Streamlit UI
st.title('Monte Carlo Simulation App')

# User input for stock symbol
stock = st.text_input('Enter stock symbol (e.g., AAPL)', value='AAPL')

# User input for length of steps (T)
T = st.slider('Number of days to simulate (T)', 50, 1000, 252)

# User input for number of simulations
sims = st.slider('Number of Simulations', 100, 5000, 1000)

# User input for confidence level
confidence_level = st.slider('Confidence Level', 0.01, 0.1, 0.05, step=0.01)

# Run simulation
start_date = '2023-01-01'
end_date = dt.today()

var, cvar, prices_matrix = monte_carlo_simulation(stock, T, sims, confidence_level)

# Display results
st.write(f"Value at Risk at {int(confidence_level*100)}% confidence level: {var:.2f}")
st.write(f"Conditional Value at Risk: {cvar:.2f}")

# Plot simulation results
fig, ax = plt.subplots(figsize=(10,5), dpi=1000)
plt.style.use('seaborn-v0_8-dark')

for i in range(prices_matrix.shape[1]):
    plt.plot(prices_matrix[:,i], alpha=0.8, linewidth=0.5)

plt.axhline(var, linestyle='--', color='red', label=f'VaR = {var:.2f}')
plt.axhline(cvar, linestyle='--', color='black', label=f'CVaR = {cvar:.2f}')
plt.title(f'Monte Carlo Simulation of {stock}', fontsize=16) 
plt.xlabel('Time (Days)', fontsize=12) 
plt.ylabel('Price ($)', fontsize=12)  
plt.legend()

st.pyplot(fig)