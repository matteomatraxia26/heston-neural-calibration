"""European calls and scalar Brent implied-volatility inversion."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def black_scholes_call(spot, strike, maturity, volatility, r=0.045, q=0.012):
    """Scalar call price; zero volatility returns discounted intrinsic value."""
    values = [spot, strike, maturity, volatility, r, q]
    if not np.isfinite(values).all() or min(spot, strike, maturity) <= 0 or volatility < 0:
        raise ValueError("Require finite inputs, S/K/T > 0 and volatility >= 0")
    forward = spot * np.exp((r - q) * maturity)
    discount = np.exp(-r * maturity)
    if volatility == 0:
        return float(discount * max(forward - strike, 0))
    w = volatility * np.sqrt(maturity)
    d1 = np.log(forward / strike) / w + w / 2
    return float(discount * (forward * ndtr(d1) - strike * ndtr(d1 - w)))


def implied_volatility(price, spot, strike, maturity, r=0.045, q=0.012):
    """Return decimal IV; invalid prices raise, intrinsic price returns zero.

    The upper price bound has no finite IV. No out-of-bounds price is clipped.
    """
    lower = black_scholes_call(spot, strike, maturity, 0.0, r, q)
    upper = spot * np.exp(-q * maturity)
    if not np.isfinite(price) or price < lower or price >= upper:
        raise ValueError("Call price outside finite-IV no-arbitrage bounds")
    if price == lower:
        return 0.0
    # Normalisation makes root accuracy independent of spot's currency units.
    forward = spot * np.exp((r - q) * maturity)
    k, c = strike / forward, price / upper
    objective = lambda sigma: black_scholes_call(1.0, k, maturity, sigma, 0, 0) - c
    ceiling = 1.0
    while objective(ceiling) < 0 and ceiling < 128:
        ceiling *= 2
    if objective(ceiling) < 0:
        raise ValueError("Could not bracket a finite implied volatility")
    return float(brentq(objective, 0, ceiling, xtol=1e-12, rtol=1e-12))
