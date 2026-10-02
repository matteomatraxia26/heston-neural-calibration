# Supplied SPX option quotes

`spx_options.csv` is a minimal transformation of
`Fast-Calibration-of-Heston-Model/data/SPX_data_26.02.24.csv` from the supplied
reference archive. All 4,277 rows are retained in their original order.

| Column | Definition |
|---|---|
| `strike` | Original strike |
| `bid`, `ask` | Original option quotes |
| `spot` | `(underlying_bid + underlying_ask) / 2` |
| `time_to_maturity` | Original supplied year fraction, unchanged |
| `option_type` | Original call/put code; loader retains `C` only |

No IV labels, neural predictions or inferred calendar dates are stored in this file.
`load_market_data()` recomputes mid prices and IV using the repository's Brent solver.
It checks finite positive inputs, ordered quotes, price bounds and the training domain
`0.8 <= K/S <= 1.2`, `0.1 <= T <= 2.0`, leaving 2,130 calls at 14 maturities.
There is no bid-ask weighting, spread filter or selection by fit residual.

Flat continuously compounded assumptions are `r=4.5%`, `q=1.2%` per year,
consistent with the main archive's documented market experiment. They are not
market curves extracted from this snapshot. Labels and calibration use the same carry.

The filename suggests `26.02.24`; the reference notebook describes February 26,
2024; every supplied `quote_datetime` instead reads `2026-02-24 16:15:00`.
The original supplied maturity is also one day below the naive calendar-day
difference divided by 365. Settlement conventions were not supplied and are not
reconstructed. Consequently the project refers only to a **supplied snapshot of SPX
option quotes** and preserves the supplied maturities. Its source date and vendor
provenance have not been independently verified.
