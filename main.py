from Option_pricer import OptionPricer
import numpy as np

pricer = OptionPricer(
    S0=100,
    K=105,
    r=0.05,
    sigma=0.25,
    T=1.0,
    N=252,
    M=100000
)

call_price = pricer.price_call()
put_price = pricer.price_put()
american_put = pricer.price_american_put()

print(f"Price of the European call: {call_price:.4f}")
print(f"Price of the European put: {put_price:.4f}")
print(f"Price of the American put: {american_put:.4f}")

parity_check = call_price - put_price
theoretical = 100 - 105 * np.exp(-0.05)
print(
    f"Put-Call Parity: Actual={parity_check:.4f}, Theoretical={theoretical:.4f}")

pricer.plot_distribution()
pricer.plot_convergence()

pricer.plot_exercise_boundary()
