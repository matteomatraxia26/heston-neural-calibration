"""Five-parameter box-constrained least-squares calibration via SciPy."""
import time

import numpy as np
from scipy.optimize import minimize

from .heston import BOUNDS, PARAMETER_NAMES


def features(params, market_data):
    return np.column_stack([np.tile(params, (len(market_data), 1)),
                            market_data.moneyness, market_data.time_to_maturity])


def calibration_objective(params, market_data, model):
    residual = model.predict(features(params, market_data)) - market_data.implied_vol.to_numpy()
    return float(np.mean(residual**2))


def calibrate_heston(market_data, model, initial_guess=None):
    """One start, finite-difference L-BFGS-B; no network Jacobians.

    Optimise scaled parameters on [0,1]^5 to balance unlike parameter units.
    Return the unscaled MSE and the optimiser's actual convergence status.
    """
    columns = ["moneyness", "time_to_maturity", "implied_vol"]
    if len(market_data) < 6 or not np.isfinite(market_data[columns]).all().all():
        raise ValueError("At least six finite observations required")
    if (market_data[columns] <= 0).any().any():
        raise ValueError("Moneyness, maturity and observed IV must be positive")
    lo, hi = BOUNDS.T
    initial = np.array([1.8, .045, .35, -.65, .035] if initial_guess is None else initial_guess)
    if initial.shape != (5,) or not np.isfinite(initial).all() or np.any(initial < lo) or np.any(initial > hi):
        raise ValueError("Initial guess outside parameter bounds")
    started = time.perf_counter()
    result = minimize(lambda z: calibration_objective(lo + z * (hi - lo), market_data, model),
                      (initial - lo) / (hi - lo), method="L-BFGS-B", bounds=[(0, 1)] * 5,
                      options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9, "maxls": 40})
    params = lo + result.x * (hi - lo)
    return {"parameters": dict(zip(PARAMETER_NAMES, map(float, params))),
            "objective": float(result.fun), "success": bool(result.success),
            "iterations": int(result.nit), "message": str(result.message),
            "elapsed_seconds": time.perf_counter() - started,
            "feller_margin": float(2 * params[0] * params[1] - params[2]**2)}
