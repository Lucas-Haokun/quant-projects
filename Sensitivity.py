import numpy as np
import matplotlib.pyplot as plt
from Option_pricer import OptionPricer


def run_sensitivity(param_name, values, fixed_params):

    # Vary one parameter while holding all others fixed
    print(f"\n===== Sensitivity Analysis: {param_name} =====")
    results = []

    for v in values:
        params = fixed_params.copy()
        params[param_name] = v

        pricer = OptionPricer(**params)
        price = pricer.price_american_put()
        results.append((v, price))

        print(f"{param_name} = {v:.2f}  ->  American Put Price = {price:.4f}")

    # Plot price vs parameter
    plt.figure(figsize=(10, 5))
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    plt.plot(xs, ys, marker='o', linewidth=2, color='blue')
    plt.title(f'American Put Price vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Option Price')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_boundaries(param_name, values, fixed_params):
    plt.figure(figsize=(10, 5))

    for v in values:
        params = fixed_params.copy()
        params[param_name] = v

        pricer = OptionPricer(**params)
        pricer.simulate()

        paths = pricer.price_paths
        M, N = paths.shape
        cashflow = np.maximum(pricer.K - paths[:, -1], 0)
        exercise_time = np.full(M, N - 1)

        for t in range(N - 2, 0, -1):
            S_t = paths[:, t]
            intrinsic = np.maximum(pricer.K - S_t, 0)
            itm = intrinsic > 0
            if np.sum(itm) > 0:
                cashflow_discounted = cashflow[itm] * np.exp(
                    -pricer.r * pricer.dt * (exercise_time[itm] - t)
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

        plt.plot(boundary_times, boundary_prices,
                 linewidth=2, label=f'{param_name} = {v}')

    plt.title(f'Optimal Exercise Boundary vs {param_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price at Exercise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":

    # Base parameters: S0=100, K=105, r=0.05, sigma=0.25, T=1.0
    base_params = dict(S0=100, K=105, r=0.05,
                       sigma=0.25, T=1.0, N=252, M=50000)

    # Run sensitivity analysis for each parameter
    run_sensitivity('sigma', [0.10, 0.20, 0.30, 0.40], base_params)
    plot_boundaries('sigma', [0.10, 0.20, 0.30, 0.40], base_params)

    run_sensitivity('K', [95, 100, 105, 110], base_params)
    plot_boundaries('K', [95, 100, 105, 110], base_params)

    run_sensitivity('T', [0.5, 1.0, 1.5, 2.0], base_params)
    plot_boundaries('T', [0.5, 1.0, 1.5, 2.0], base_params)

    run_sensitivity('r', [0.01, 0.03, 0.05, 0.07], base_params)
    plot_boundaries('r', [0.01, 0.03, 0.05, 0.07], base_params)
