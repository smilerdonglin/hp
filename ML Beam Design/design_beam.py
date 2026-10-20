"""Inverse beam-shape design via the trained surrogate.

Design goal (mirrors the paper's experimental findings, now as a target):
  minimize the trailing-edge thermal gradient G (and with it the cooling
  rate) while (a) keeping melt width at least that of the 60 um Gaussian
  reference and (b) keeping peak temperature below boiling (conduction
  mode, low keyhole/spatter propensity).

Stages:
  1. Gradient-based optimization THROUGH the surrogate (softmax over 49
     spot logits, Adam, multi-start) -- possible because the numpy MLP
     exposes input gradients.
  2. Exact-model polish: local random search on the winning candidates
     using the fast superposition forward model (no surrogate error).
  3. Verification of the final design with the full transient FD solver.

Run:  python design_beam.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import beam_shapes as bs
import surrogate as sg
import thermal_model as tm
from generate_dataset import pattern_kernel, forward, all_metrics, ALL_METRIC_NAMES

OUT = Path(__file__).parent / "outputs"

# objective weights (loss is in physical units, roughly balanced)
PEN_WIDTH = 50.0    # per um of width deficit, squared
PEN_PEAK = 0.05     # per K above boiling, squared
PEAK_MIN_C = 1800.0  # required superheat: guards against the optimizer
PEN_PEAK_LOW = 0.5   # "gaming" G by barely melting (lack-of-fusion risk)
PEN_KEYHOLE = 2000.0  # per unit of Hmax/hs above the 8.65 keyhole threshold,
                      # squared (Gruenewald AIME 2025 shape-collapsed scaling)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def loss_terms(m, width_target, cfg):
    """m = metric vector (physical units). Returns scalarized loss parts."""
    # Objective rides on G only: the cooling rate has a very narrow dynamic
    # range in the quasi-steady frame (~v-dominated) and is reported, not
    # optimized.
    peak, width, G, hmax_hs = m[0], m[1], m[4], m[7]
    l_obj = G
    l_w = PEN_WIDTH * max(width_target - width, 0.0) ** 2
    l_p = PEN_PEAK * max(peak - cfg.boil_c, 0.0) ** 2
    l_p += PEN_PEAK_LOW * max(PEAK_MIN_C - peak, 0.0) ** 2
    l_k = PEN_KEYHOLE * max(hmax_hs - bs.HMAX_HS_KEYHOLE, 0.0) ** 2
    return l_obj + l_w + l_p + l_k


def main() -> None:
    cfg = tm.Config()
    rng = np.random.default_rng(7)
    model, extra = sg.MLP.load(OUT / "surrogate.npz")
    y_mean, y_std = extra["y_mean"], extra["y_std"]

    green = tm.greens_field(cfg, bs.W_SPOT_M)
    green_ref = tm.greens_field(cfg, bs.W_GAUSS_REF_M)

    def exact_metrics(fr):
        return all_metrics(green, fr, cfg)

    # --- baseline: 60 um Gaussian reference ---
    m_gauss = np.concatenate([
        tm.metrics(tm.compose(green_ref, np.array([1.0]),
                              np.array([[0.0, 0.0]]), cfg), cfg),
        bs.gaussian_beam_metrics(cfg, bs.W_GAUSS_REF_M)])
    width_target = m_gauss[1]
    print(f"Gaussian reference: width = {width_target:.0f} um, "
          f"peak = {m_gauss[0]:.0f} C, G = {m_gauss[4]:.0f} K/mm, "
          f"cooling = {m_gauss[5]:.0f} K/ms, Hmax/hs = {m_gauss[7]:.2f}")

    # --- stage 1: gradient descent through the surrogate, multi-start ---
    print("\nStage 1: surrogate gradient optimization (48 starts x 400 steps)")
    n_starts, n_steps, lr = 48, 400, 0.05
    candidates = []
    for s in range(n_starts):
        z = rng.normal(0, 0.5, bs.N_SPOTS)
        m1 = np.zeros_like(z); v1 = np.zeros_like(z)
        for it in range(1, n_steps + 1):
            fr = softmax(z)
            pred = (model(fr[None, :])[0] * y_std) + y_mean
            peak, width = pred[0], pred[1]
            # d(loss)/d(pred) in physical units
            dL = np.zeros(len(pred))
            dL[4] = 1.0
            if width < width_target:
                dL[1] = -2 * PEN_WIDTH * (width_target - width)
            if peak > cfg.boil_c:
                dL[0] = 2 * PEN_PEAK * (peak - cfg.boil_c)
            elif peak < PEAK_MIN_C:
                dL[0] = -2 * PEN_PEAK_LOW * (PEAK_MIN_C - peak)
            if pred[7] > bs.HMAX_HS_KEYHOLE:
                dL[7] = 2 * PEN_KEYHOLE * (pred[7] - bs.HMAX_HS_KEYHOLE)
            # surrogate outputs are standardized -> chain rule via y_std
            dfr = model.input_gradient(fr[None, :], (dL * y_std)[None, :])[0]
            # softmax jacobian: dz = fr * (dfr - fr.dfr)
            dz = fr * (dfr - float(fr @ dfr))
            m1 = 0.9 * m1 + 0.1 * dz
            v1 = 0.999 * v1 + 0.001 * dz**2
            z -= lr * m1 / (np.sqrt(v1) + 1e-8)
        fr = softmax(z)
        candidates.append((loss_terms(exact_metrics(fr), width_target, cfg), z))
    candidates.sort(key=lambda c: c[0])
    print(f"  best exact loss after stage 1: {candidates[0][0]:.1f}")

    # --- stage 2: exact-model polish, annealed local search on top-5 ---
    print("Stage 2: exact-model local search (top 5 candidates x 1500 iters)")
    best_loss, best_z = candidates[0]
    for l0, z0 in candidates[:5]:
        z, l = z0.copy(), l0
        for it in range(1500):
            sigma = 0.4 * (1 - it / 1500) + 0.05  # annealed step size
            z_try = z + rng.normal(0, sigma, bs.N_SPOTS)
            l_try = loss_terms(exact_metrics(softmax(z_try)), width_target, cfg)
            if l_try < l:
                z, l = z_try, l_try
        if l < best_loss:
            best_loss, best_z = l, z
    fr_best = softmax(best_z)
    # prune spots that carry <1% power (hardware-friendlier), renormalize
    fr_best = bs.normalize(np.where(fr_best > 0.01, fr_best, 0.0))
    m_best = exact_metrics(fr_best)
    print(f"  final exact loss: {loss_terms(m_best, width_target, cfg):.1f}, "
          f"active spots: {(fr_best > 0).sum()}")

    # --- stage 3: full FD verification of the designed shape ---
    print("Stage 3: full transient FD verification of the designed beam...")
    temp_fd, _, _ = tm.run_fd(cfg, pattern_kernel(cfg, fr_best))
    m_fd = np.concatenate([tm.metrics(temp_fd, cfg),
                           bs.beam_metrics(fr_best, cfg)])

    # --- report ---
    rows = {
        "gaussian60 (ref)": m_gauss,
        "ring preset": exact_metrics(bs.preset("ring")),
        "line preset": exact_metrics(bs.preset("line")),
        "ML-designed (fast fwd)": m_best,
        "ML-designed (FD verify)": m_fd,
    }
    lines = ["shape," + ",".join(ALL_METRIC_NAMES)]
    print(f"\n  {'shape':>24} | " + " | ".join(f"{n:>15}" for n in ALL_METRIC_NAMES))
    for k, v in rows.items():
        print(f"  {k:>24} | " + " | ".join(f"{x:15.2f}" for x in v))
        lines.append(k.replace(",", ";") + "," + ",".join(f"{x:.3f}" for x in v))
    (OUT / "design_metrics.csv").write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(OUT / "designed_beam.npz", fracs=fr_best,
                        offsets_m=bs.spot_offsets_m(), metrics_fd=m_fd,
                        metric_names=np.array(ALL_METRIC_NAMES))

    dG = (m_fd[4] - m_gauss[4]) / m_gauss[4]
    dC = (m_fd[5] - m_gauss[5]) / m_gauss[5]
    dW = (m_fd[1] - m_gauss[1]) / m_gauss[1]
    print(f"\nDesigned vs Gaussian (FD-verified): G {dG:+.1%}, "
          f"cooling rate {dC:+.1%}, melt width {dW:+.1%}, "
          f"peak {m_fd[0]:.0f} C (boiling {cfg.boil_c:.0f} C), "
          f"Hmax/hs {m_fd[7]:.2f} vs keyhole threshold {bs.HMAX_HS_KEYHOLE}")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = tm.grid(cfg)
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    f_gauss = tm.compose(green_ref, np.array([1.0]), np.array([[0.0, 0.0]]), cfg)
    f_ring = forward(green, bs.preset("ring"), cfg)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    ax = axes[0, 0]
    im = ax.imshow(fr_best.reshape(bs.N_SIDE, bs.N_SIDE), cmap="inferno", origin="lower")
    ax.set_title("ML-designed spot pattern (scan ->, +x is right)")
    ax.set_xlabel("spot column (50 um pitch)")
    plt.colorbar(im, ax=ax, label="power fraction")
    for ax, fld, title in [(axes[0, 1], temp_fd, "designed beam — FD verified"),
                           (axes[1, 0], f_gauss, "Gaussian 60 um reference"),
                           (axes[1, 1], f_ring, "ring preset")]:
        im = ax.imshow(fld, cmap="inferno", origin="lower", extent=ext,
                       vmin=cfg.ambient_c, vmax=max(3000.0, float(temp_fd.max())))
        ax.contour(fld, levels=[cfg.melt_c], colors="cyan", linewidths=1,
                   extent=ext, origin="lower")
        ax.set_title(title + " (cyan = melt boundary)", fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="T (C)")
    fig.tight_layout()
    fig.savefig(OUT / "designed_beam.png", dpi=150)
    print(f"Saved: {OUT / 'designed_beam.png'}, design_metrics.csv, designed_beam.npz")


if __name__ == "__main__":
    main()
