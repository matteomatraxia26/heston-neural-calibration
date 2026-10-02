"""Real PyTorch training and inference; one self-contained state-dict checkpoint."""
from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn

from .data import DOMAIN, FEATURE_NAMES, ROOT
from .heston import R, Q


def error_metrics(actual, predicted):
    error = (np.asarray(predicted) - np.asarray(actual)) * 1e4
    return {"mae_iv_bp": float(np.mean(abs(error))),
            "rmse_iv_bp": float(np.sqrt(np.mean(error**2))),
            "max_iv_bp": float(np.max(abs(error)))}


class HestonNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 128), nn.Tanh(), nn.Linear(128, 128),
                                    nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.register_buffer("input_mean", torch.zeros(7, dtype=torch.float64))
        self.register_buffer("input_std", torch.ones(7, dtype=torch.float64))
        # Float64 keeps SciPy's small finite-difference steps observable.
        self.double()

    def forward(self, x):
        """Predict log(decimal IV); scaler is fitted only on training rows."""
        return self.layers((x - self.input_mean) / self.input_std).squeeze(-1)

    @torch.inference_mode()
    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != 7 or len(x) == 0 or not np.isfinite(x).all()
                or np.any(x < DOMAIN[:, 0]) or np.any(x > DOMAIN[:, 1])):
            raise ValueError("Expected nonempty in-domain (n, 7) features")
        # Conversion and scaling are intentionally inside the timed method.
        prediction = self(torch.as_tensor(x, dtype=torch.float64)).exp().numpy()
        if not np.isfinite(prediction).all() or np.any(prediction <= 0):
            raise ArithmeticError("Invalid neural IV prediction")
        return prediction


def train_model(x, y, epochs=600, patience=60, seed=42):
    """80/10/10 split, Adam, log-IV MSE, validation-only early stopping."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or x.shape[1] != 7 or y.shape != (len(x),) or len(x) < 100
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0)
            or np.any(x < DOMAIN[:, 0]) or np.any(x > DOMAIN[:, 1])):
        raise ValueError("Need >=100 valid in-domain features and positive IV labels")
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    order = np.random.default_rng(seed).permutation(len(x))
    train, validation, test = np.split(order, [int(.8 * len(x)), int(.9 * len(x))])
    model = HestonNN()
    model.input_mean.copy_(torch.from_numpy(x[train].mean(axis=0)))
    model.input_std.copy_(torch.from_numpy(x[train].std(axis=0)))
    xt, yt = torch.from_numpy(x), torch.from_numpy(np.log(y))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=.5, min_lr=1e-5)
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.from_numpy(train)[torch.randperm(len(train))].split(512):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = nn.functional.mse_loss(model(xt[validation]), yt[validation]).item()
        if not np.isfinite(val_loss):
            raise ArithmeticError("Nonfinite training loss")
        scheduler.step(val_loss)
        history.append(val_loss)
        if val_loss < best:
            best, stale, best_epoch = val_loss, 0, epoch
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: validation log-IV MSE {val_loss:.8g}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    report = {"seed": seed, "epochs_run": epoch, "best_epoch": best_epoch,
              "training_seconds": time.perf_counter() - started,
              "split_counts": [len(train), len(validation), len(test)],
              "validation": error_metrics(y[validation], model.predict(x[validation])),
              "test": error_metrics(y[test], model.predict(x[test])), "validation_loss": history}
    return model, report


def save_model(model, report, path=ROOT / "models/heston_nn.pt"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "features": list(FEATURE_NAMES),
                "domain": DOMAIN.tolist(), "r": R, "q": Q, "report": report}, path)


def load_model(path=ROOT / "models/heston_nn.pt"):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/train_model.py first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["features"] != list(FEATURE_NAMES) or checkpoint["domain"] != DOMAIN.tolist()
            or checkpoint["r"] != R or checkpoint["q"] != Q):
        raise ValueError("Checkpoint features, domain or carry differ from this code")
    model = HestonNN()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["report"]
