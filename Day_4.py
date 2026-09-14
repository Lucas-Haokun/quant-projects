import numpy as np


S0 = 100
K = 105
r = 0.05
sigma = 0.25
T = 1.0
N = 252
dt = T / N
M = 100000

np.random.seed(42)


Z = np.random.normal(0, 1, (M, N))

daily_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

log_returns = np.cumsum(daily_returns, axis=1)
price_paths = S0 * np.exp(log_returns)


S_T = price_paths[:, -1]
payoffs = np.maximum(S_T - K, 0)
option_price = np.exp(-r * T) * np.mean(payoffs)

print(f"Option price: {option_price:.4f}")
