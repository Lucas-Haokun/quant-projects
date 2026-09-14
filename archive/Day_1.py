import numpy as np
import matplotlib.pyplot as plt


S_0 = 100
mu = 0.05
sigma = 0.2
T = 1.0
N = 252
dt = T / N


np.random.seed(42)
Z = np.random.normal(0, 1, N)
daily_returns = sigma * np.sqrt(dt) * Z
price_path = S_0 * np.exp(np.cumsum(daily_returns))


plt.figure(figsize=(10, 5))
plt.plot(price_path)
plt.title('Random Walk Simulation (No Drift)')
plt.xlabel('Trading Days (0 to 252)')
plt.ylabel('Price')
plt.grid(True)
plt.show()
