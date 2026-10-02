"""Stable Heston P1/P2 inversion, adapted from the original pricing.py."""
from functools import lru_cache
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import roots_legendre

from .black_scholes import implied_volatility

PARAMETER_NAMES = ("kappa", "theta", "xi", "rho", "v0")
BOUNDS = np.array([(0.1, 5.0), (0.005, 0.15), (0.05, 1.5),
                   (-0.95, -0.05), (0.005, 0.15)])
R, Q = 0.045, 0.012  # Flat assumptions shared by labels, market IV and calibration.


def _validate(params, maturity):
    p, t = np.asarray(params, float), np.asarray(maturity, float)
    if p.shape[-1:] != (5,) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Five finite Heston parameters and finite maturity required")
    if np.any(p[..., [0, 1, 2, 4]] <= 0) or np.any(abs(p[..., 3]) >= 1) or np.any(t <= 0):
        raise ValueError("Require kappa/theta/xi/v0/T > 0 and |rho| < 1")
    return p, t


def heston_characteristic_function(u, params, maturity, j=2):
    """CF of log(S_T/F), under cash (j=2) or share (j=1) measure.

    Use the square-root branch with nonnegative real part and exp(-d*T).
    Rationalisation and log1p/expm1 preserve the original numerical stability.
    Scalar parameters work with scalar or vector u; batches use u on the last axis.
    """
    p, t = _validate(params, maturity)
    if j not in (1, 2):
        raise ValueError("j must be 1 or 2")
    kappa, theta, xi, rho, v0 = np.moveaxis(p, -1, 0)
    u = np.asarray(u)
    b = kappa - (rho * xi if j == 1 else 0)
    z = b - rho * xi * 1j * u
    a = u * u + (1j * u if j == 2 else -1j * u)
    d = np.sqrt(z * z + xi * xi * a)
    zm = -xi * xi * a / (z + d)
    g = zm / (z + d)
    decay = np.exp(-d * t)
    c = kappa * theta / xi**2 * (zm * t - 2 * (np.log1p(-g * decay) - np.log1p(-g)))
    dv = zm / xi**2 * (-np.expm1(-d * t)) / (1 - g * decay)
    return np.exp(c + dv * v0)


def heston_call_price(spot, strike, maturity, params, r=R, q=Q):
    """Scalar reference: adaptive integration over [0, infinity), no COS.

    Price = S exp(-qT) P1 - K exp(-rT) P2. Integration failures raise.
    """
    _validate(params, maturity)
    if not np.isfinite([spot, strike, r, q]).all() or min(spot, strike) <= 0:
        raise ValueError("Invalid contract")
    k = strike / (spot * np.exp((r - q) * maturity))
    probabilities = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        for j in (1, 2):
            def integrand(u):
                cf = heston_characteristic_function(u, params, maturity, j)
                return float(np.real(np.exp(-1j * u * np.log(k)) * cf / (1j * u)))
            integral, error = quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-9, limit=300)
            if error > 1e-7:
                raise ArithmeticError("Integration error exceeds tolerance")
            probabilities.append(0.5 + integral / np.pi)
    c = probabilities[0] - k * probabilities[1]
    lower = max(1 - k, 0)
    if not np.isfinite(c) or c < lower - 1e-9 or c > 1 + 1e-9:
        raise ArithmeticError("Integrated call violates price bounds")
    return float(spot * np.exp(-q * maturity) * np.clip(c, lower, 1))


def heston_implied_volatility(spot, strike, maturity, params, r=R, q=Q):
    price = heston_call_price(spot, strike, maturity, params, r, q)
    return implied_volatility(price, spot, strike, maturity, r, q)


@lru_cache(maxsize=2)
def _quadrature(nodes, cutoff):
    x, w = roots_legendre(nodes)
    return (x + 1) * cutoff / 2, w * cutoff / 2


def batch_call_prices(x, nodes=384, cutoff=400.0):
    """Same P1/P2 formula, fixed Gauss-Legendre integration for training.

    x contains [kappa, theta, xi, rho, v0, K/S, T]; spot is normalised to 1.
    Raw prices are returned, so the caller can reject unconverged labels.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all() or np.any(x[:, 5] <= 0):
        raise ValueError("Expected finite (n, 7) features with positive moneyness")
    p, t = _validate(x[:, :5], x[:, 6])
    u, w = _quadrature(nodes, cutoff)
    k = x[:, 5] * np.exp(-(R - Q) * t)
    phase = np.exp(-1j * np.log(k[:, None]) * u) / (1j * u)
    probabilities = []
    for j in (1, 2):
        cf = heston_characteristic_function(u, p[:, None, :], t[:, None], j)
        probabilities.append(0.5 + np.sum(w * np.real(phase * cf), axis=1) / np.pi)
    return np.exp(-Q * t) * (probabilities[0] - k * probabilities[1])
