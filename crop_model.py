"""Core library for the soil analysis & crop recommendation project.

Contains: data loading/splitting, the PyTorch model, the training loop,
model save/load, prediction and a simple soil-analysis report.
Used by train.py, predict.py and the notebook.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df


def prepare_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = SEED):
    """Stratified train/val/test split. The scaler is fit on the training split only."""
    classes = sorted(df[TARGET].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].map(class_to_idx).to_numpy(dtype=np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    scale = lambda a: ((a - mean) / std).astype(np.float32)

    return {
        "X_train": scale(X_train), "y_train": y_train,
        "X_val": scale(X_val), "y_val": y_val,
        "X_test": scale(X_test), "y_test": y_test,
        "mean": mean, "std": std, "classes": classes,
    }


def make_loader(X, y, batch_size=64, shuffle=False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class CropNet(nn.Module):
    """Small MLP: 7 soil/climate features -> probability over crops."""

    def __init__(self, n_features: int = len(FEATURES), n_classes: int = 22,
                 hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, device, epochs=300, lr=2e-3,
                weight_decay=1e-4, patience=30, verbose=True):
    """Adam + ReduceLROnPlateau + early stopping on validation loss.
    Returns the model loaded with the best weights and a history dict."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(run_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d} | train loss {run_loss / n:.4f} acc {correct / n:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}")
        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_all(model, X, device):
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------
def save_bundle(path, model, data, hidden=(128, 64), dropout=0.2):
    bundle = {
        "state_dict": model.state_dict(),
        "mean": data["mean"].tolist(),
        "std": data["std"].tolist(),
        "classes": list(data["classes"]),
        "features": FEATURES,
        "hidden": list(hidden),
        "dropout": dropout,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_bundle(path, device=None):
    device = device or torch.device("cpu")
    b = torch.load(path, map_location=device, weights_only=True)
    model = CropNet(len(b["features"]), len(b["classes"]), tuple(b["hidden"]), b["dropout"])
    model.load_state_dict(b["state_dict"])
    model.to(device).eval()
    return model, b


# --------------------------------------------------------------------------
# Inference + soil analysis
# --------------------------------------------------------------------------
def recommend(model, bundle, sample: dict, top_k: int = 3, device=None):
    """sample: dict with keys N,P,K,temperature,humidity,ph,rainfall -> [(crop, prob), ...]"""
    device = device or torch.device("cpu")
    x = np.array([[float(sample[f]) for f in bundle["features"]]], dtype=np.float32)
    x = (x - np.array(bundle["mean"], dtype=np.float32)) / np.array(bundle["std"], dtype=np.float32)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x.astype(np.float32)).to(device)), 1)[0].cpu().numpy()
    top = probs.argsort()[::-1][:top_k]
    return [(bundle["classes"][i], float(probs[i])) for i in top]


def classify_ph(ph: float) -> str:
    if ph < 5.5:
        return "strongly acidic"
    if ph < 6.5:
        return "slightly acidic"
    if ph <= 7.5:
        return "neutral"
    if ph <= 8.5:
        return "alkaline"
    return "strongly alkaline"


def soil_report(sample: dict, df: pd.DataFrame, crop: str) -> list[str]:
    """Compare the soil/climate values to the typical range (10th-90th percentile)
    of the recommended crop in the training data."""
    sub = df[df[TARGET] == crop]
    lines = [f"Soil pH {sample['ph']:.1f}: {classify_ph(sample['ph'])}."]
    for f in FEATURES:
        lo, hi = sub[f].quantile(0.10), sub[f].quantile(0.90)
        v = float(sample[f])
        status = "within" if lo <= v <= hi else ("below" if v < lo else "above")
        lines.append(f"{f}: {v:.1f} is {status} the typical range for {crop} ({lo:.1f}-{hi:.1f}).")
    return lines
