"""Generate the beam-shape -> thermal-response dataset.

1. Computes (or loads from cache) the single-spotlet Green's response and
   the 60 um Gaussian reference response via full transient FD.
2. Validates the superposition forward model against direct FD for a few
   random patterns (reports metric-level relative errors).
3. Samples N random beam shapes, evaluates all of them with the fast
   superposition model, and saves X (49 power fractions) and y (6 thermal
   metrics) to dataset.npz + dataset.csv, plus a preview figure.

Run:  python generate_dataset.py [N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import beam_shapes as bs
import thermal_model as tm

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def pattern_kernel(cfg: tm.Config, fracs: np.ndarray) -> np.ndarray:
    """Absorbed-flux kernel of a fully rendered pattern (main lobes +
    side-lobe satellites + envelope droop), for direct FD runs."""
    weights, offsets = bs.render_lobes(fracs)
    weights = weights / weights.sum()
    half = int(np.ceil((150e-6 + bs.SL_OFFSET_M + 3 * bs.W_SPOT_M) / cfg.dx))
    ax = np.arange(-half, half + 1) * cfg.dx
    kx, ky = np.meshgrid(ax, ax)
    kern = np.zeros_like(kx)
    for w, (ox, oy) in zip(weights, offsets):
        g = np.exp(-2 * ((kx - ox) ** 2 + (ky - oy) ** 2) / bs.W_SPOT_M**2)
        kern += w * g / (g.sum() * cfg.dx**2)
    return kern * cfg.absorptivity * cfg.power_w


def forward(green: dict, fracs: np.ndarray, cfg: tm.Config) -> np.ndarray:
    """Fast forward model of a rendered pattern via superposition."""
    weights, offsets = bs.render_lobes(fracs)
    return tm.compose(green, weights, offsets, cfg)


ALL_METRIC_NAMES = tm.METRIC_NAMES + bs.BEAM_METRIC_NAMES


def all_metrics(green: dict, fracs: np.ndarray, cfg: tm.Config) -> np.ndarray:
    """6 thermal metrics + [Phi, Hmax/hs] for a commanded pattern."""
    thermal = tm.metrics(forward(green, fracs, cfg), cfg)
    return np.concatenate([thermal, bs.beam_metrics(fracs, cfg)])


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg = tm.Config()
    rng = np.random.default_rng(42)

    print("Computing/loading Green's responses (full FD, cached)...")
    t0 = time.time()
    green = tm.greens_field(cfg, bs.W_SPOT_M)
    green_ref = tm.greens_field(cfg, bs.W_GAUSS_REF_M)
    print(f"  done in {time.time() - t0:.1f} s")

    # ---- validation: superposition vs direct FD ----
    print("Validating superposition against direct FD (3 rendered patterns)...")
    worst = 0.0
    for i in range(3):
        fr = bs.random_pattern(rng)
        t0 = time.time()
        temp_fd, _, _ = tm.run_fd(cfg, pattern_kernel(cfg, fr))
        m_fd = tm.metrics(temp_fd, cfg)
        m_fast = tm.metrics(forward(green, fr, cfg), cfg)
        rel = np.abs(m_fast - m_fd) / np.maximum(np.abs(m_fd), 1e-9)
        worst = max(worst, float(rel[:4].max()))  # geometry + peak metrics
        print(f"  pattern {i}: FD {time.time() - t0:.1f} s | "
              f"peak/width/length/area rel.err = "
              + ", ".join(f"{r:.3%}" for r in rel[:4]))
    print(f"  worst geometric relative error: {worst:.3%}")

    # ---- baselines ----
    base_rows = {}
    base_rows["gaussian60"] = np.concatenate([
        tm.metrics(tm.compose(green_ref, np.array([1.0]),
                              np.array([[0.0, 0.0]]), cfg), cfg),
        bs.gaussian_beam_metrics(cfg, bs.W_GAUSS_REF_M)])
    for name in ["center", "ring", "line", "leading_arc", "trailing_arc"]:
        base_rows[name] = all_metrics(green, bs.preset(name), cfg)
    print("\nBaseline responses (P = %.0f W, v = %.0f mm/s):" % (cfg.power_w, cfg.speed_m_s * 1e3))
    print(f"  {'shape':>13} | " + " | ".join(f"{n:>15}" for n in ALL_METRIC_NAMES))
    for k, v in base_rows.items():
        print(f"  {k:>13} | " + " | ".join(f"{x:15.2f}" for x in v))
    print(f"  (regime thresholds: Hmax/hs {bs.HMAX_HS_TRANSITION} = "
          f"conduction->transition, {bs.HMAX_HS_KEYHOLE} = transition->keyhole)")

    # ---- dataset ----
    print(f"\nGenerating {n_samples} samples with the fast forward model...")
    t0 = time.time()
    X = np.zeros((n_samples, bs.N_SPOTS))
    Y = np.zeros((n_samples, len(ALL_METRIC_NAMES)))
    for i in range(n_samples):
        fr = bs.random_pattern(rng)
        X[i] = fr
        Y[i] = all_metrics(green, fr, cfg)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples}  ({time.time() - t0:.1f} s)")
    melted = (Y[:, 1] > 0).mean()
    print(f"  fraction of samples that melt: {melted:.1%}")

    np.savez_compressed(OUT / "dataset.npz", X=X, Y=Y,
                        metric_names=np.array(ALL_METRIC_NAMES))
    with (OUT / "dataset.csv").open("w", encoding="utf-8") as f:
        f.write(",".join([f"a{i}" for i in range(bs.N_SPOTS)] + ALL_METRIC_NAMES) + "\n")
        for xi, yi in zip(X, Y):
            f.write(",".join(f"{v:.6g}" for v in np.concatenate([xi, yi])) + "\n")

    # ---- preview figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    for ax, i in zip(axes.ravel(), rng.choice(n_samples, 18, replace=False)):
        ax.imshow(X[i].reshape(bs.N_SIDE, bs.N_SIDE), cmap="inferno", origin="lower")
        ax.set_title(f"w={Y[i, 1]:.0f} um, G={Y[i, 4]:.0f} K/mm", fontsize=8)
        ax.axis("off")
    fig.suptitle("Random CBC spot patterns (scan direction: +x = right)")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_preview.png", dpi=150)
    print(f"\nSaved: {OUT / 'dataset.npz'}, dataset.csv, dataset_preview.png")


if __name__ == "__main__":
    main()
