"""Ten focused regression tests; unittest also runs the numerical subset."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.black_scholes import black_scholes_call, implied_volatility
from src.heston import heston_call_price, heston_implied_volatility, batch_call_prices
from src.data import ROOT, generate_dataset, load_market_data


class NumericalTests(unittest.TestCase):
    def test_black_scholes_round_trip(self):
        for strike, maturity, vol in [(90, .5, .25), (110, 2, .4), (100, 1, 2.0)]:
            price = black_scholes_call(100, strike, maturity, vol)
            self.assertAlmostEqual(implied_volatility(price, 100, strike, maturity), vol, places=10)
        self.assertAlmostEqual(black_scholes_call(100, 100, 1, .2, .03, 0), 9.413403383853016)

    def test_invalid_inputs_and_boundaries(self):
        for bad in [-1, 100, np.nan]:
            with self.assertRaises(ValueError):
                implied_volatility(bad, 100, 100, 1, 0, 0)
        self.assertEqual(implied_volatility(10, 110, 100, 1, 0, 0), 0)
        for maturity, volatility in [(0, .2), (1, -.2)]:
            with self.assertRaises(ValueError):
                black_scholes_call(100, 100, maturity, volatility)
        with self.assertRaises(ValueError):
            heston_call_price(100, 100, 1, [2, .04, -.3, -.7, .04])

    def test_heston_price_and_small_xi_limit(self):
        params = [2, .04, .3, -.7, .04]
        prices = [heston_call_price(100, k, 1, params) for k in [80, 90, 100, 110, 120]]
        self.assertTrue(np.isfinite(prices).all())
        self.assertTrue(np.all(np.diff(prices) < 0))
        self.assertTrue(np.all(np.diff(prices, n=2) > 0))
        for k, price in zip([80, 90, 100, 110, 120], prices):
            self.assertGreaterEqual(price, black_scholes_call(100, k, 1, 0))
            self.assertLess(price, 100 * np.exp(-.012))
        low_xi = heston_call_price(100, 100, 1, [2, .04, .0001, -.7, .04], .03, 0)
        self.assertLess(abs(low_xi - black_scholes_call(100, 100, 1, .2, .03, 0)), .001)

    def test_original_price_regressions(self):
        # Fixed regression values supplied in the main archive's pricing tests.
        # No external pricing package is needed at runtime.
        price = heston_call_price(1, 1.05, .24927764054195672,
                                  [3.16, .09, .4, -.2, .1], .0225, .02)
        self.assertAlmostEqual(price, .0404774515, delta=1e-8)
        price = heston_call_price(.9792189645694596, 1, .7,
                                  [1.2, .08, 1.8, -.45, .09], .05, .02)
        self.assertAlmostEqual(price, .0641016, delta=1e-6)

    def test_batch_labels_against_adaptive_integration(self):
        x, y, report = generate_dataset(32, seed=17)
        direct = np.array([heston_implied_volatility(1, row[5], row[6], row[:5]) for row in x])
        np.testing.assert_allclose(y, direct, atol=2e-5, rtol=0)
        x2, y2, _ = generate_dataset(32, seed=17)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        self.assertGreaterEqual(report["attempted"], 32)

    def test_market_cleaning(self):
        row = dict(strike=100, spot=100, bid=8, ask=9, time_to_maturity=1, option_type="C")
        rows = [row, {**row, "option_type": "P"}, {**row, "bid": 0},
                {**row, "ask": 7}, {**row, "strike": 200}, {**row, "time_to_maturity": 0},
                {**row, "bid": 100, "ask": 101}, {**row, "spot": np.nan}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quotes.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            clean = load_market_data(path)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.mid.iloc[0], 8.5)
        self.assertTrue(np.isfinite(clean.implied_vol).all())
        self.assertGreater(len(load_market_data()), 100)


class NeuralTests(unittest.TestCase):
    def test_checkpoint_prediction(self):
        from src.neural_network import load_model
        model, report = load_model()
        prediction = model.predict([[1.8, .045, .35, -.65, .035, 1, 1]])
        self.assertTrue(np.isfinite(prediction).all() and (prediction > 0).all())
        self.assertTrue(np.isfinite(list(report["test"].values())).all())
        with self.assertRaises(ValueError):
            model.predict([[1.8, .045, .35, -.65, .035, 2, 1]])

    def test_objective_discriminates_parameters(self):
        from src.calibration import calibration_objective
        from src.neural_network import load_model
        model, _ = load_model()
        truth = [1.8, .045, .35, -.65, .035]
        m, t = np.meshgrid([.9, 1, 1.1], [.25, 1, 2])
        d = pd.DataFrame({"moneyness": m.ravel(), "time_to_maturity": t.ravel()})
        d["implied_vol"] = [heston_implied_volatility(1, k, maturity, truth) for k, maturity in zip(m.ravel(), t.ravel())]
        good = calibration_objective(truth, d, model)
        bad = calibration_objective([1, .14, 1, -.2, .14], d, model)
        self.assertTrue(np.isfinite(good))
        self.assertLess(good, bad)

    def test_training_scaler_and_checkpoint_round_trip(self):
        from src.neural_network import train_model, save_model, load_model
        x, y, _ = generate_dataset(100, seed=3)
        model, report = train_model(x, y, epochs=2)
        train = np.random.default_rng(42).permutation(len(x))[:80]
        np.testing.assert_allclose(model.input_mean.numpy(), x[train].mean(axis=0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            save_model(model, report, path)
            loaded, _ = load_model(path)
            np.testing.assert_allclose(model.predict(x), loaded.predict(x), atol=0, rtol=0)

    def test_calibration_moves_and_respects_bounds(self):
        from src.calibration import calibrate_heston, calibration_objective, features
        from src.neural_network import load_model
        from src.heston import BOUNDS
        model, _ = load_model()
        m, t = np.meshgrid([.9, 1, 1.1], [.25, 1, 2])
        d = pd.DataFrame({"moneyness": m.ravel(), "time_to_maturity": t.ravel()})
        d["implied_vol"] = model.predict(features([2, .04, .4, -.7, .04], d))
        initial = [1, .07, .7, -.4, .07]
        initial_loss = calibration_objective(initial, d, model)
        result = calibrate_heston(d, model, initial)
        params = np.array(list(result["parameters"].values()))
        self.assertTrue(result["success"], result["message"])
        self.assertLess(result["objective"], initial_loss / 100)
        self.assertTrue(np.all(params >= BOUNDS[:, 0]) and np.all(params <= BOUNDS[:, 1]))


if __name__ == "__main__":
    unittest.main()
