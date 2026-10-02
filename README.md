# Heston Calibration with Neural Networks

Calibrate the Heston stochastic volatility model to SPX option quotes using a
**PyTorch implied-volatility surrogate** and **SciPy constrained optimisation**.
The project connects characteristic-function pricing, numerical IV inversion,
supervised learning and market calibration in five focused Python modules.

**A trained checkpoint and an executed notebook are included. No retraining is required.**

[Explore the notebook](notebooks/heston_calibration.ipynb) · [Run the project](#quick-start) · [Data assumptions](data/README.md)

## Results at a glance

| Measurement | Recorded result |
|---|---:|
| Surrogate test MAE / RMSE | **6.33 / 8.86 IV bp** |
| SPX in-sample fit MAE / RMSE | **27.89 / 38.70 IV bp** |
| Numerical Heston IV evaluation | **7.37 ms per option** |
| Neural IV evaluation, batch of 200 | **0.00237 ms per option** |
| Batched IV evaluation speedup versus scalar adaptive integration | **3,104×** |

One IV basis point is `0.0001` in decimal volatility. Synthetic test metrics use
5,000 held-out observations; the SPX fit uses 2,130 calls across 14 maturities.
The speedup compares **Heston integration + Brent inversion** with **batched neural
inference**, including preprocessing. It is not a full-calibration speedup or a
comparison against an optimised vectorised numerical pricer.

![Observed SPX IV versus neural IV at calibrated Heston parameters](figures/spx_fit.png)

*Each point is a fitted SPX quote; the dashed line represents a perfect fit.
The fit is in-sample, under the documented rate and dividend assumptions.*

## Why use a surrogate?

A calibration algorithm evaluates many candidate Heston parameter vectors. Each
candidate requires an implied volatility for every option: integrate the Heston
characteristic function to obtain a price, then invert Black–Scholes numerically.
A trained neural network approximates this mapping with a fast forward pass.

The network predicts **IV from parameters and a contract**. SciPy then searches
for the five Heston parameters that best match the observed IVs.

1. **Generate labels:** sample Heston parameters, moneyness and maturity; compute call prices and invert them with Brent.
2. **Train offline:** learn the seven-input mapping to log-IV using PyTorch.
3. **Calibrate online:** evaluate candidate parameters in batches and minimise IV error with L-BFGS-B.
4. **Check the result:** compare the fitted surface with SPX quotes and reprice selected contracts numerically.

## Quick start

From a terminal on macOS or Linux:

```bash
git clone https://github.com/matteomatraxia26/heston-neural-calibration.git
cd heston-neural-calibration
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python scripts/benchmark.py
jupyter notebook notebooks/heston_calibration.ipynb
```

The benchmark and notebook load `models/heston_nn.pt`. They run on CPU without a GPU.
On Windows, activate the virtual environment with `.venv\Scripts\Activate.ps1` in PowerShell.

Optional full retraining: `python scripts/train_model.py`. This regenerates the
synthetic labels and **overwrites the checkpoint**. To execute the notebook from the terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/heston_calibration.ipynb --inplace
```

## Model and numerical pricing

Under the risk-neutral measure:

$$dS_t=(r-q)S_t\,dt+\sqrt{v_t}S_t\,dW_t^S,$$
$$dv_t=\kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW_t^v,\qquad d\langle W^S,W^v\rangle_t=\rho\,dt.$$

| Parameter | Interpretation | Training and calibration bounds |
|---|---|---|
| `kappa` | Mean-reversion speed | 0.1–5.0 |
| `theta` | Long-run variance | 0.005–0.15 |
| `xi` | Volatility of variance | 0.05–1.5 |
| `rho` | Spot/variance correlation | −0.95–−0.05 |
| `v0` | Initial variance | 0.005–0.15 |

European calls use `C = S exp(-qT) P1 - K exp(-rT) P2`, with P1/P2 obtained by
integrating the characteristic function. The scalar reference uses SciPy adaptive
quadrature over the infinite interval. Label generation uses vectorised
Gauss–Legendre quadrature of the same formula, checked at two resolutions.
Unstable and low-vega labels are rejected. Every accepted label uses Brent IV inversion.

## Neural surrogate and calibration

The mapping is `(kappa, theta, xi, rho, v0, K/S, T) → log(IV)`.
The MLP has three 128-unit tanh hidden layers; exponentiation produces positive IV.

Training uses 50,000 accepted numerical labels, an 80/10/10 split, training-only
input standardisation, Adam and validation-based checkpoint selection with early
stopping. Seeds are fixed. Float64 preserves the small input perturbations used
by SciPy's finite differences. The checkpoint includes weights and scaler buffers.

The domain is `0.8 ≤ K/S ≤ 1.2` and `0.1 ≤ T ≤ 2.0` years. Parameters and moneyness
are sampled uniformly; maturity is log-uniform. The checkpoint assumes fixed
`r = 4.5%` and `q = 1.2%`; varying carry is not supported by the seven-input interface.

Calibration minimises mean squared error between neural IV and observed mid-price IV.
L-BFGS-B uses the training bounds, numerical differences, one initial guess and a
linear parameter rescaling to a unit box. No analytic network Jacobian is supplied.

| Parameter | SPX estimate |
|---|---:|
| kappa | 4.712692 |
| theta | 0.049075 |
| xi | 1.500000 |
| rho | −0.773514 |
| v0 | 0.037475 |

The fit converged in 30 iterations, taking 0.87 s in the recorded run. **`xi` reaches
its upper bound:** this is a converged solution within the trained box; global
optimality is not established. The Feller margin is −1.78745. Feller positivity is
reported rather than imposed; its failure does not invalidate Heston pricing.

## Validation and benchmark details

**10 tests passed; all 11 notebook code cells executed without errors.** Tests cover
IV inversion, numerical pricing, label consistency, quote cleaning, checkpoint
loading, training-only scaling and bounded optimisation.

The notebook also demonstrates synthetic parameter recovery (IV fit RMSE 2.61 bp).
On 30 SPX quotes repriced at the fitted parameters, numerical Heston versus market
RMSE is 60.53 bp; neural versus numerical Heston RMSE is 9.85 bp. This subset differs
from the full fitted sample. Synthetic test maximum error is 108.92 bp, so average
accuracy should not be interpreted as a worst-case guarantee.

<details>
<summary>Timing protocol, secondary run and recorded environment</summary>

The primary benchmark is the standalone script: the same 200 seeded contracts and
one fixed Heston parameter vector are evaluated by both methods. After warm-up,
reference timing uses three repetitions; neural timing uses nine groups of 100
batches. Reported times are medians, divided by the batch size.

Neural timing includes input checks, tensor conversion, standardisation, forward
pass, exponentiation and output conversion. Label generation, training and loading
the checkpoint are excluded. Benchmark prediction MAE/RMSE is 3.80/4.80 IV bp.

A second run in the notebook measured 3,978× (9.49 ms versus 0.00238 ms per option).
The table uses the smaller standalone result; both runs illustrate timing variability.
Offline label generation took 9.83 s and training 78.63 s. Training completed 600
epochs and restored epoch 594, selected using validation loss.

Recorded local environment: macOS 26.5.1 arm64, Python 3.14.4, NumPy 2.5.3,
SciPy 1.18.1, pandas 2.3.3, PyTorch 2.14.1, Matplotlib 3.11.2 and pytest 9.1.1.
Inference uses CPU, one PyTorch thread and float64. Timings and training results
can vary across hardware and software environments.

</details>

## Repository map

| Path | Purpose |
|---|---|
| `src/black_scholes.py` | European calls and Brent IV inversion |
| `src/heston.py` | Stable characteristic function and numerical integration |
| `src/data.py` | Synthetic labels and SPX quote cleaning |
| `src/neural_network.py` | PyTorch training, inference and checkpoint handling |
| `src/calibration.py` | Five-parameter bounded SciPy optimisation |
| `scripts/` | Training and CPU benchmark entry points |
| `notebooks/heston_calibration.ipynb` | Executed walkthrough and calibration results |
| `tests/test_core.py` | Ten focused tests |
| `data/`, `models/`, `figures/` | SPX quotes, trained checkpoint and fit plot |

## Limitations and data source

- One supplied SPX snapshot is fitted in-sample, using assumed flat carry. Source dates and maturity conventions remain unresolved; see [the data notes](data/README.md).
- The active `xi` bound restricts the fit. A wider training domain would require new labels and retraining; one start does not establish a global optimum.
- Label rejection changes the accepted distribution. Accuracy outside the training domain is unsupported, and the neural surface is not constrained to be arbitrage-free.
- The speedup is specific to the scalar adaptive reference and batch size. It does not establish the same gain over vectorised quadrature or for an entire calibration.

The SPX data is a reduced extract of `SPX_data_26.02.24.csv` from the
`Fast-Calibration-of-Heston-Model` reference project, which also inspired the linear
workflow. The stable numerical routines were refactored from the earlier
Heston Calibration Lab implementation. The current PyTorch checkpoint was trained
on newly generated numerical labels, not on weights from the reference project.
