import numpy as np
import matplotlib.pyplot as plt


class OptionPricer:

    def __init__(self, S0, K, r, sigma, T, N=252, M=100000, seed=42):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.M = M
        self.seed = seed
        self.dt = T / N
        self.price_paths = None
        self.S_T = None

    def simulate(self):
        np.random.seed(self.seed)

        Z = np.random.normal(0, 1, (self.M, self.N))

        # Ito correction: risk-neutral log-drift is (r - 0.5 * sigma^2)
        daily_returns = (self.r - 0.5 * self.sigma**2) * \
            self.dt + self.sigma * np.sqrt(self.dt) * Z

        # Cumulative log-returns: additive in log space, then exponentiated back to prices
        log_returns = np.cumsum(daily_returns, axis=1)
        self.price_paths = self.S0 * np.exp(log_returns)

        self.S_T = self.price_paths[:, -1]

    def price_call(self):
        if self.S_T is None:
            self.simulate()

        payoffs = np.maximum(self.S_T - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price

    def price_put(self):
        if self.S_T is None:
            self.simulate()

        payoffs = np.maximum(self.K - self.S_T, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price

    def price_american_put(self):
        if self.price_paths is None:
            self.simulate()
        paths = self.price_paths
        M, N = paths.shape
        cashflow = np.maximum(self.K - paths[:, -1], 0)
        exercise_time = np.full(M, N - 1)
        for t in range(N - 2, 0, -1):
            S_t = paths[:, t]
            intrinsic = np.maximum(self.K - S_t, 0)
            itm = intrinsic > 0

            # Only in-the-money paths matter for exercise decision
            if np.sum(itm) > 0:
                cashflow_discounted = cashflow[itm] * \
                    np.exp(-self.r * self.dt * (exercise_time[itm] - t))
                X = np.column_stack([
                    np.ones(np.sum(itm)),
                    S_t[itm],
                    S_t[itm] ** 2
                ])

        # OLS regression to estimate continuation value conditional on S_t
        beta = np.linalg.lstsq(X, cashflow_discounted, rcond=None)[0]
        continuation = X @ beta
        exercise = intrinsic[itm] > continuation
        itm_indices = np.where(itm)[0]
        exercise_indices = itm_indices[exercise]
        cashflow[exercise_indices] = intrinsic[exercise_indices]
        exercise_time[exercise_indices] = t
        american_price = np.mean(
            cashflow * np.exp(-self.r * self.dt * exercise_time))
        return american_price

    def plot_exercise_boundary(self):
        if self.price_paths is None:
            self.simulate()
        paths = self.price_paths
        M, N = paths.shape
        cashflow = np.maximum(self.K - paths[:, -1], 0)
        exercise_time = np.full(M, N - 1)
        for t in range(N - 2, 0, -1):
            S_t = paths[:, t]
            intrinsic = np.maximum(self.K - S_t, 0)
            itm = intrinsic > 0
            if np.sum(itm) > 0:
                cashflow_discounted = cashflow[itm] * np.exp(
                    -self.r * self.dt * (exercise_time[itm] - t)
                )
                X = np.column_stack([
                    np.ones(np.sum(itm)),
                    S_t[itm],
                    S_t[itm] ** 2
                ])
                beta = np.linalg.lstsq(X, cashflow_discounted, rcond=None)[0]
                continuation = X @ beta
                exercise = intrinsic[itm] > continuation
                itm_indices = np.where(itm)[0]
                exercise_indices = itm_indices[exercise]
                cashflow[exercise_indices] = intrinsic[exercise_indices]
                exercise_time[exercise_indices] = t
        boundary_times = []
        boundary_prices = []
        for t in range(1, N):
            exercised = exercise_time == t
            if np.sum(exercised) > 0:
                price_95 = np.percentile(paths[exercised, t], 95)
                boundary_times.append(t)
                boundary_prices.append(price_95)
        plt.figure(figsize=(10, 5))
        plt.plot(boundary_times, boundary_prices, color='red', linewidth=2)
        plt.axhline(self.K, color='black', linestyle='--',
                    linewidth=1.5, label=f'Strike K = {self.K}')
        plt.title('Optimal Exercise Boundary for American Put (95th Percentile)')
        plt.xlabel('Time Step')
        plt.ylabel('Stock Price at Exercise')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_distribution(self):
        if self.S_T is None:
            self.simulate()

        plt.figure(figsize=(10, 5))
        plt.hist(self.S_T, bins=100, alpha=0.7,
                 color='skyblue', edgecolor='black')
        plt.axvline(self.K, color='red', linestyle='--',
                    linewidth=2, label=f'Strike K = {self.K}')
        plt.axvline(np.mean(self.S_T), color='green', linestyle='-',
                    linewidth=2, label=f'Mean = {np.mean(self.S_T):.2f}')
        plt.title(f'Distribution of Terminal Prices (M={self.M}, N={self.N})')
        plt.xlabel('Price at Maturity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_convergence(self):

        if self.S_T is None:
            self.simulate()

        payoffs = np.maximum(self.S_T - self.K, 0)
        cumulative_mean = np.cumsum(payoffs) / np.arange(1, self.M + 1)
        cumulative_price = np.exp(-self.r * self.T) * cumulative_mean

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_price, color='blue', linewidth=1.5)
        plt.axhline(cumulative_price[-1], color='red', linestyle='--', linewidth=2,
                    label=f'Final Price = {cumulative_price[-1]:.4f}')
        plt.title('Convergence of Monte Carlo Option Price')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
