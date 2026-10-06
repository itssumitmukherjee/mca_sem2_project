"""Train the crop recommendation model.

Usage:  python train.py [--csv data/Crop_recommendation.csv] [--out artifacts]
Outputs (in --out): crop_model.pt, metrics.json, training_curves.png, confusion_matrix.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import crop_model as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cm.set_seed()
    device = cm.get_device()
    print("Device:", device)

    df = cm.load_dataframe(args.csv)
    data = cm.prepare_data(df)
    print({k: len(data[k]) for k in ("y_train", "y_val", "y_test")})

    train_loader = cm.make_loader(data["X_train"], data["y_train"], args.batch_size, shuffle=True)
    val_loader = cm.make_loader(data["X_val"], data["y_val"], args.batch_size)

    model = cm.CropNet(n_classes=len(data["classes"]))
    model, hist = cm.train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    y_pred, _ = cm.predict_all(model, data["X_test"], device)
    y_true = data["y_test"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nTest accuracy: {acc:.4f} | macro-F1: {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=data["classes"], digits=3))

    cm.save_bundle(out / "crop_model.pt", model, data)
    (out / "metrics.json").write_text(json.dumps(
        {"test_accuracy": acc, "test_macro_f1": f1, "epochs_run": len(hist["train_loss"])}, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Greens",
                xticklabels=data["classes"], yticklabels=data["classes"], ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)
    print("Saved model and plots to", out.resolve())


if __name__ == "__main__":
    main()
