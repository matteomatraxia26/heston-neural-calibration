"""Seven-feature synthetic labels and deliberately simple SPX cleaning."""
from pathlib import Path

import numpy as np
import pandas as pd

from .black_scholes import implied_volatility
from .heston import BOUNDS, R, Q, batch_call_prices

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ("kappa", "theta", "xi", "rho", "v0", "moneyness", "maturity")
DOMAIN = np.vstack([BOUNDS, (0.8, 1.2), (0.1, 2.0)])


def generate_dataset(n=50000, seed=42):
    """Uniform parameters/moneyness, log-uniform maturity; return n valid rows.

    Compare two resolutions of the same integration formula. Reject low-vega
    labels because tiny price errors can otherwise become large IV errors.
    No target is produced by a neural network or by the reference ZIP.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    rows, targets, attempted = [], [], 0
    while len(rows) < n:
        if attempted >= 3 * n + 256:
            raise ArithmeticError("Too many rejected labels; inspect the domain")
        x = rng.uniform(DOMAIN[:, 0], DOMAIN[:, 1], size=(min(256, n - len(rows)), 7))
        x[:, 6] = np.exp(rng.uniform(*np.log(DOMAIN[6]), len(x)))
        coarse = batch_call_prices(x)
        fine = batch_call_prices(x, nodes=768, cutoff=800.0)
        attempted += len(x)
        for row, c1, c2 in zip(x, coarse, fine):
            if not np.isfinite([c1, c2]).all():
                continue
            try:
                iv = implied_volatility(c2, 1, row[5], row[6], R, Q)
            except ValueError:
                continue
            if iv <= 0:
                continue
            d1 = (-np.log(row[5]) + (R - Q + iv**2 / 2) * row[6]) / (iv * np.sqrt(row[6]))
            vega = np.exp(-Q * row[6] - d1**2 / 2) * np.sqrt(row[6] / (2 * np.pi))
            if vega < 1e-4 or abs(c1 - c2) / vega > 2e-5:
                continue
            rows.append(row)
            targets.append(iv)
    return np.array(rows), np.array(targets), {"seed": seed, "attempted": attempted, "accepted": n}


def load_market_data(path=ROOT / "data/spx_options.csv"):
    """Read supplied quotes, compute mids and invert our Black-Scholes function."""
    d = pd.read_csv(path)
    columns = ["strike", "bid", "ask", "spot", "time_to_maturity"]
    for col in columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    valid = (np.isfinite(d[columns]).all(axis=1) & (d[columns] > 0).all(axis=1)
             & (d.ask >= d.bid) & d.option_type.eq("C"))
    d = d.loc[valid].copy()
    d["moneyness"] = d.strike / d.spot
    d["mid"] = (d.bid + d.ask) / 2
    d = d[d.moneyness.between(*DOMAIN[5]) & d.time_to_maturity.between(*DOMAIN[6])].copy()
    ivs = []
    for row in d.itertuples():
        try:
            ivs.append(implied_volatility(row.mid, row.spot, row.strike, row.time_to_maturity, R, Q))
        except ValueError:
            ivs.append(np.nan)
    d["implied_vol"] = ivs
    d = d[np.isfinite(d.implied_vol) & (d.implied_vol > 0)].reset_index(drop=True)
    if d.empty:
        raise ValueError("No usable in-domain call quotes")
    return d
