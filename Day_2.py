import numpy as np
import matplotlib.pyplot as plt


S0 = 100
mu = 0.08
sigma = 0.25
T = 1.0
N = 252
dt = T / N
M = 20


np.random.seed(42)


Z = np.random.normal(0, 1, (M, N))
daily_returns = mu * dt + sigma * np.sqrt(dt) * Z
log_returns = np.cumsum(daily_returns, axis=1)
price_paths = S0 * np.exp(log_returns)


plt.figure(figsize=(12, 6))

for i in range(M):
    plt.plot(price_paths[i, :], linewidth=0.8, alpha=0.6)


mean_path = np.mean(price_paths, axis=0)
plt.plot(mean_path, linewidth=2.5, color='black', label='Mean Path')
plt.title(f'Geometric Brownian Motion ({M} Paths, mu={mu}, sigma={sigma})')
plt.xlabel('Trading Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
