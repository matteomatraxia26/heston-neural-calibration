"""Regenerate numerical labels and train the CPU PyTorch checkpoint."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_dataset
from src.neural_network import train_model, save_model


if __name__ == "__main__":
    started = time.perf_counter()
    x, y, generation = generate_dataset(n=50000, seed=42)
    generation["seconds"] = time.perf_counter() - started
    print("Generated labels:", generation, flush=True)
    model, report = train_model(x, y)
    report["generation"] = generation
    save_model(model, report)
    print("Held-out test errors:", report["test"])
    print("Saved models/heston_nn.pt")
