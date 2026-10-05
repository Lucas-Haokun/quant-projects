"""Validate the Monte Carlo pricer against independent benchmarks.

Benchmarks:
  - Black-Scholes closed form for the European put
  - Cox-Ross-Rubinstein (CRR) binomial tree for the American put
"""
import math
import numpy as np
from Option_pricer import OptionPricer


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put(S0, K, r, sigma, T):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)


def crr_put(S0, K, r, sigma, T, steps=5000, american=True):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    values = np.maximum(K - S0 * u ** (steps - j) * d ** j, 0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, K - S0 * u ** (i - j) * d ** j)
    return values[0]


if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 105, 0.05, 0.25, 1.0
    pricer = OptionPricer(S0, K, r, sigma, T, N=252, M=100000)

    mc_euro = pricer.price_put()
    mc_amer = pricer.price_american_put()
    bs_euro = black_scholes_put(S0, K, r, sigma, T)
    crr_amer = crr_put(S0, K, r, sigma, T)

    print(f"European put : MC = {mc_euro:.4f} | Black-Scholes = {bs_euro:.4f} "
          f"| error = {abs(mc_euro - bs_euro) / bs_euro:.2%}")
    print(f"American put : LSM = {mc_amer:.4f} | CRR tree (5000 steps) = {crr_amer:.4f} "
          f"| error = {abs(mc_amer - crr_amer) / crr_amer:.2%}")
    print(f"Early exercise premium (CRR): {crr_amer - bs_euro:.4f}")
