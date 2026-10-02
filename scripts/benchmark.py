"""CPU IV evaluation timing: adaptive Heston + Brent versus batched PyTorch."""
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy

from src.heston import heston_implied_volatility


def benchmark_inputs():
    rng = np.random.default_rng(2026)
    params = np.array([1.8, .045, .35, -.65, .035])
    return np.column_stack([np.tile(params, (200, 1)), rng.uniform(.8, 1.2, 200),
                            np.exp(rng.uniform(np.log(.1), np.log(2), 200))])


def median_seconds(function, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        times.append(time.perf_counter() - started)
    return float(np.median(times)), value


def run_benchmark():
    import torch
    from src.neural_network import load_model, error_metrics
    torch.set_num_threads(1)
    model, _ = load_model()
    x = benchmark_inputs()
    direct = lambda: np.array([heston_implied_volatility(1, row[5], row[6], row[:5]) for row in x])
    for _ in range(10):
        model.predict(x)
    heston_implied_volatility(1, x[0, 5], x[0, 6], x[0, :5])
    exact_seconds, exact = median_seconds(direct, 3)
    # Time groups to keep clock resolution negligible relative to inference.
    def neural_group():
        for _ in range(100):
            prediction = model.predict(x)
        return prediction
    neural_seconds, prediction = median_seconds(neural_group, 9)
    neural_seconds /= 100
    result = {"options": len(x), "exact_ms_per_option": exact_seconds * 1000 / len(x),
              "nn_ms_per_option": neural_seconds * 1000 / len(x),
              "speedup": exact_seconds / neural_seconds, "accuracy": error_metrics(exact, prediction),
              "python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__),
              "torch_threads": torch.get_num_threads(), "dtype": "float64"}
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Batched throughput: scaling, tensor conversion and exp included; training excluded.")
    print("Reference = adaptive P1/P2 integration + Brent, not a complete calibration.")
    return result


if __name__ == "__main__":
    run_benchmark()
