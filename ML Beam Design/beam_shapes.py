"""CBC/OPA beam parameterization with hardware-realism rendering.

A beam shape is a 7x7 grid of dwell positions (50 um pitch, spanning
+/-150 um) with nonnegative dwell/power fractions -- a design vector of
49 numbers. Physically (Weber et al., JMMP 9:85, 2025) a DBL shape is a
time-sequence of ONE moving main lobe, and the per-cell "power fraction"
is a dwell-time fraction; in the quasi-static regime (shape refresh <<
melt-pool response ~ms, Sawannia 2024) the melt pool sees the
dwell-weighted time-averaged intensity modeled here.

Rendering model (per Weber et al. scaling laws):
- Spotlet 1/e^2 radius 40 um -> main-lobe diameter 80 um, which makes
  the +/-150 um grid exactly span the point-placement field
  (x_set,max = 1.9 * d_MainLobe).
- Each commanded position also deposits FOUR side-lobe satellites
  (square-array symmetry) at +/-SL_OFFSET along x and y, with ~6% of the
  main-lobe peak when the position is centered.
- A Gaussian field envelope (d_Env = 6.7 * d_MainLobe) attenuates
  off-center lobes: a main lobe steered to the field edge drops to ~53%
  peak, matching Weber's reported 54%.
The composed pattern is renormalized to total laser power (the ~16% of
power in the diffuse higher-order background is folded back in).

The scan direction is +x. Column index increases toward the leading
(front) side of the melt pool.
"""
from __future__ import annotations

import numpy as np

N_SIDE = 7
PITCH_M = 50e-6
W_SPOT_M = 40e-6       # 1/e^2 radius of one spotlet (main lobe)
W_GAUSS_REF_M = 60e-6  # single-beam Gaussian reference (original script)

N_SPOTS = N_SIDE * N_SIDE

# Hardware-realism render constants (Weber et al. 2025 scaling)
D_MAIN_M = 2 * W_SPOT_M            # main-lobe 1/e^2 diameter
SL_OFFSET_M = 1.9 * D_MAIN_M       # side-lobe spacing ~= placement half-field
D_ENV_M = 6.7 * D_MAIN_M           # single-fiber Gaussian envelope (1/e^2 dia)
SL_PEAK_CENTERED = 0.06            # satellite peak / main peak, centered cell


def envelope(r_m: np.ndarray | float) -> np.ndarray | float:
    """Field-envelope attenuation at radius r from the optical axis."""
    return np.exp(-8.0 * np.asarray(r_m) ** 2 / D_ENV_M**2)


# Intrinsic satellite strength, calibrated so a centered cell's satellites
# have SL_PEAK_CENTERED relative peak after envelope attenuation.
_SAT_REL = SL_PEAK_CENTERED / float(envelope(SL_OFFSET_M))


def render_lobes(fracs: np.ndarray):
    """Expand commanded dwell fractions into physical lobes.

    Returns (weights, offsets_m): for every active cell, the
    envelope-attenuated main lobe plus its four side-lobe satellites
    (satellites steer WITH the main lobe -- all diffraction orders shift
    together under a linear phase tilt). Weights are relative; the
    composer renormalizes to total power.
    """
    fracs = np.clip(np.asarray(fracs, dtype=float).ravel(), 0.0, None)
    cells = spot_offsets_m()
    sat_dirs = np.array([[SL_OFFSET_M, 0], [-SL_OFFSET_M, 0],
                         [0, SL_OFFSET_M], [0, -SL_OFFSET_M]])
    weights, offsets = [], []
    for a, pos in zip(fracs, cells):
        if a < 1e-12:
            continue
        weights.append(a * float(envelope(np.hypot(*pos))))
        offsets.append(pos)
        for d in sat_dirs:
            sat = pos + d
            weights.append(a * _SAT_REL * float(envelope(np.hypot(*sat))))
            offsets.append(sat)
    return np.array(weights), np.array(offsets)


def spot_offsets_m() -> np.ndarray:
    """(49, 2) array of (x, y) offsets of each spotlet from beam center."""
    c = np.arange(N_SIDE) - (N_SIDE - 1) / 2
    ox, oy = np.meshgrid(c * PITCH_M, c * PITCH_M)
    return np.column_stack([ox.ravel(), oy.ravel()])


def normalize(a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=float).ravel(), 0.0, None)
    s = a.sum()
    return a / s if s > 0 else a


def _radius_grid() -> np.ndarray:
    off = spot_offsets_m()
    return np.hypot(off[:, 0], off[:, 1])


def preset(name: str) -> np.ndarray:
    """Named amplitude patterns (power fractions, sum = 1)."""
    off = spot_offsets_m()
    r = _radius_grid()
    a = np.zeros(N_SPOTS)
    if name == "center":
        a[N_SPOTS // 2] = 1.0
    elif name == "ring":
        a[(r > 75e-6) & (r < 125e-6)] = 1.0
    elif name == "line":  # transverse line through center (x ~ 0)
        a[np.abs(off[:, 0]) < 1e-9] = 1.0
    elif name == "leading_arc":  # arc opening backward, on the front side
        ang = np.arctan2(off[:, 1], off[:, 0])
        a[(r > 75e-6) & (r < 160e-6) & (np.abs(ang) < np.pi / 3)] = 1.0
    elif name == "trailing_arc":
        ang = np.arctan2(off[:, 1], off[:, 0])
        a[(r > 75e-6) & (r < 160e-6) & (np.abs(ang) > 2 * np.pi / 3)] = 1.0
    else:
        raise ValueError(f"unknown preset '{name}'")
    return normalize(a)


BEAM_METRIC_NAMES = ["phi_peak_to_mean", "Hmax_hs"]

# Melting-mode thresholds from Gruenewald et al., AIME 10:100164 (2025):
# conduction->transition at Hmax/hs = 6.25 +/- 0.85,
# transition->keyhole at Hmax/hs = 8.65 +/- 0.30 (all beam shapes collapse).
HMAX_HS_TRANSITION = 6.25
HMAX_HS_KEYHOLE = 8.65


def _enthalpy_norm(cfg, two_w0_m: float) -> float:
    """Dimensionless enthalpy H/hs (Hann/King form incl. latent heat) for
    the configured power/speed and a 2nd-moment spot diameter 2w0."""
    h_melt = cfg.cp * (cfg.melt_c - cfg.ambient_c) + cfg.latent_heat
    return (cfg.absorptivity * cfg.power_w
            / (cfg.rho * h_melt * np.sqrt(cfg.alpha * cfg.speed_m_s * two_w0_m**3)))


def beam_metrics(fracs: np.ndarray, cfg) -> np.ndarray:
    """[Phi, Hmax/hs] for a rendered CBC pattern.

    Phi = I_peak * pi * w0^2 / P with w0 the 2nd-moment beam radius
    (Phi = 2 for an ideal Gaussian). Hmax/hs = (Phi/2) * H/hs is the
    shape-corrected dimensionless enthalpy whose thresholds classify the
    melting regime independently of beam shape (Gruenewald 2025).
    """
    weights, offsets = render_lobes(fracs)
    weights = weights / weights.sum()
    # Compose the intensity map on a compact grid via kernel pasting.
    dx = 5e-6
    half = int(np.ceil((SL_OFFSET_M + 150e-6 + 3 * W_SPOT_M) / dx))
    n = 2 * half + 1
    kh = int(np.ceil(3 * W_SPOT_M / dx))
    ax = np.arange(-kh, kh + 1) * dx
    kx, ky = np.meshgrid(ax, ax)
    kern = np.exp(-2 * (kx**2 + ky**2) / W_SPOT_M**2)
    kern /= kern.sum() * dx**2  # unit power
    intensity = np.zeros((n, n))
    for w, (ox_m, oy_m) in zip(weights, offsets):
        ix = half + int(round(ox_m / dx))
        iy = half + int(round(oy_m / dx))
        intensity[iy - kh:iy + kh + 1, ix - kh:ix + kh + 1] += w * kern
    intensity *= cfg.power_w  # W/m^2

    total = intensity.sum() * dx**2
    ax_full = (np.arange(n) - half) * dx
    xx, yy = np.meshgrid(ax_full, ax_full)
    xbar = (intensity * xx).sum() * dx**2 / total
    ybar = (intensity * yy).sum() * dx**2 / total
    var_r = (intensity * ((xx - xbar) ** 2 + (yy - ybar) ** 2)).sum() * dx**2 / total
    w0 = float(np.sqrt(2 * var_r))  # 2nd-moment radius (= 1/e^2 for Gaussian)

    phi = float(intensity.max()) * np.pi * w0**2 / total
    hmax_hs = 0.5 * phi * _enthalpy_norm(cfg, 2 * w0)
    return np.array([phi, hmax_hs])


def gaussian_beam_metrics(cfg, w0_m: float) -> np.ndarray:
    """Beam metrics of an ideal single-mode Gaussian (Phi = 2 exactly)."""
    return np.array([2.0, _enthalpy_norm(cfg, 2 * w0_m)])


def random_pattern(rng: np.random.Generator) -> np.ndarray:
    """Random shapes for dataset coverage: mixes sparse random patterns
    with structured families (rings, arcs, lines, center+ring hybrids)."""
    off = spot_offsets_m()
    r = _radius_grid()
    ang = np.arctan2(off[:, 1], off[:, 0])
    family = rng.integers(0, 5)
    a = np.zeros(N_SPOTS)

    if family == 0:  # k random spots, random amplitudes
        k = int(rng.integers(1, 20))
        idx = rng.choice(N_SPOTS, size=k, replace=False)
        a[idx] = rng.uniform(0.2, 1.0, size=k)
    elif family == 1:  # ring of random radius/thickness
        r0 = rng.uniform(40e-6, 150e-6)
        w = rng.uniform(20e-6, 60e-6)
        a[np.abs(r - r0) < w] = 1.0
    elif family == 2:  # arc at random orientation/width
        r0 = rng.uniform(50e-6, 160e-6)
        th0 = rng.uniform(-np.pi, np.pi)
        half = rng.uniform(np.pi / 6, np.pi / 2)
        d = np.angle(np.exp(1j * (ang - th0)))
        a[(np.abs(r - r0) < 40e-6) & (np.abs(d) < half)] = 1.0
    elif family == 3:  # line at random offset along x
        xo = rng.choice(np.unique(off[:, 0]))
        a[np.abs(off[:, 0] - xo) < 1e-9] = 1.0
        a *= rng.uniform(0.3, 1.0, size=N_SPOTS)
    else:  # center spot + ring, random split
        a = preset("ring") * rng.uniform(0.2, 0.8)
        a[N_SPOTS // 2] += 1.0 - a.sum()
    # random smooth amplitude jitter so fractions are not only binary
    a *= rng.uniform(0.6, 1.0, size=N_SPOTS)
    if a.sum() <= 0:
        a[N_SPOTS // 2] = 1.0
    return normalize(a)
