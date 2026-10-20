"""Minimal dependency-free MLP surrogate (numpy only, Adam, tanh).

Maps 49 spot power fractions -> 6 standardized thermal metrics. Small on
purpose: the input is low-dimensional and the physics (for this linear
conduction model) is smooth. Swap in PyTorch later without changing the
pipeline if a high-fidelity simulator (keyhole/CFD) generates the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MLP:
    def __init__(self, sizes=(49, 128, 128, 6), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        """Returns output and per-layer activations for backprop."""
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            acts.append(h)
        return h, acts

    def __call__(self, X):
        return self.forward(X)[0]

    def backward(self, acts, dY):
        """Gradients of a loss with output-grad dY. Returns (dW, db, dX)."""
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        g = dY
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (1 - acts[i + 1] ** 2)  # tanh'
            dW[i] = acts[i].T @ g
            db[i] = g.sum(axis=0)
            g = g @ self.W[i].T
        return dW, db, g

    def input_gradient(self, X, dY):
        """d(loss)/d(input) for output-gradient dY — used in inverse design."""
        _, acts = self.forward(X)
        return self.backward(acts, dY)[2]

    def save(self, path: Path, extra: dict):
        arrs = {f"W{i}": w for i, w in enumerate(self.W)}
        arrs.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez_compressed(path, n_layers=len(self.W), **arrs, **extra)

    @classmethod
    def load(cls, path: Path):
        d = np.load(path, allow_pickle=False)
        n = int(d["n_layers"])
        m = cls.__new__(cls)
        m.W = [d[f"W{i}"] for i in range(n)]
        m.b = [d[f"b{i}"] for i in range(n)]
        extra = {k: d[k] for k in d.files
                 if not (k.startswith(("W", "b")) or k == "n_layers")}
        return m, extra


def train(model: MLP, X, Y, epochs=400, batch=128, lr=1e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    mW = [np.zeros_like(w) for w in model.W]; vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]; vb = [np.zeros_like(b) for b in model.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(X)
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], Y[idx]
            pred, acts = model.forward(xb)
            err = pred - yb
            tot += float((err**2).sum())
            dW, db, _ = model.backward(acts, 2 * err / len(xb))
            step += 1
            for i in range(len(model.W)):
                for P, g, m_, v_ in ((model.W[i], dW[i], mW, vW),
                                     (model.b[i], db[i], mb, vb)):
                    m_[i] = b1 * m_[i] + (1 - b1) * g
                    v_[i] = b2 * v_[i] + (1 - b2) * g**2
                    P -= lr * (m_[i] / (1 - b1**step)) / (np.sqrt(v_[i] / (1 - b2**step)) + eps)
        if (ep + 1) % 50 == 0:
            log(f"  epoch {ep + 1}/{epochs}  train MSE (std units) = {tot / n:.5f}")
    return model
