"""Train the neural surrogate on the generated dataset.

Run:  python train_surrogate.py
Outputs: outputs/surrogate.npz (weights + output scaler), parity plot,
and per-metric R^2 on a held-out validation split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import surrogate as sg
import thermal_model as tm

OUT = Path(__file__).parent / "outputs"


def main() -> None:
    d = np.load(OUT / "dataset.npz")
    X, Y = d["X"], d["Y"]
    names = list(d["metric_names"])

    # Standardize outputs (inputs are already fractions in [0, 1], sum 1)
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-12
    Ys = (Y - y_mean) / y_std

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 10)
    vi, ti = idx[:n_val], idx[n_val:]

    model = sg.MLP(sizes=(X.shape[1], 128, 128, Y.shape[1]), seed=0)
    print(f"Training on {len(ti)} samples, validating on {len(vi)}...")
    sg.train(model, X[ti], Ys[ti], epochs=1200, batch=128, lr=1e-3)

    pred = model(X[vi]) * y_std + y_mean
    true = Y[vi]
    print("\nValidation R^2 per metric:")
    for j, name in enumerate(names):
        ss_res = ((pred[:, j] - true[:, j]) ** 2).sum()
        ss_tot = ((true[:, j] - true[:, j].mean()) ** 2).sum() + 1e-12
        print(f"  {str(name):>18}: {1 - ss_res / ss_tot:.4f}")

    model.save(OUT / "surrogate.npz", {"y_mean": y_mean, "y_std": y_std})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for j, (ax, name) in enumerate(zip(axes.ravel(), names)):
        ax.scatter(true[:, j], pred[:, j], s=6, alpha=0.4)
        lim = [true[:, j].min(), true[:, j].max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("simulated"); ax.set_ylabel("surrogate")
        ax.set_title(str(name), fontsize=10)
    fig.suptitle("Surrogate parity — held-out validation set")
    fig.tight_layout()
    fig.savefig(OUT / "surrogate_parity.png", dpi=150)
    print(f"\nSaved: {OUT / 'surrogate.npz'}, surrogate_parity.png")


if __name__ == "__main__":
    main()
