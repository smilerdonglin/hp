"""Thermal forward model for CBC beam-shape design.

2D thin-plate transient conduction model of a moving laser on 316L,
refactored from ../simulate_laser_scanning_thermal.py so the heat source
can be an arbitrary intensity map (e.g., a Civan CBC/OPA spot pattern).

Because the conduction model is linear, the quasi-steady temperature
field of ANY spot pattern equals the superposition of shifted copies of
a single-spotlet response (a numerically computed Green's function).
The Green's function is computed once by full transient FD (~15 s) and
cached; after that, evaluating a new beam shape takes milliseconds.
This is exact for the linear model up to grid-shift interpolation.

Model limitations (state these in the paper): no keyhole physics, fluid
flow, latent heat, vaporization, or powder effects. Peak temperature vs.
boiling is used only as a keyhole-propensity proxy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    # Material: AISI 316L (same effective values as the original script)
    absorptivity: float = 0.35
    rho: float = 7900.0          # kg/m^3
    cp: float = 500.0            # J/(kg K)
    k: float = 16.0              # W/(m K)
    thickness_m: float = 0.20e-3  # effective 2D layer thickness
    ambient_c: float = 25.0
    melt_c: float = 1375.0
    boil_c: float = 2900.0       # keyhole-propensity proxy threshold
    latent_heat: float = 260e3   # J/kg (Mills 2002, for H/hs enthalpy norm)

    # Process
    power_w: float = 300.0
    speed_m_s: float = 0.400

    # Domain / numerics
    lx: float = 2.4e-3
    ly: float = 1.2e-3
    dx: float = 5.0e-6
    total_t: float = 3.5e-3
    x_start: float = -0.85e-3

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)

    @property
    def nx(self) -> int:
        return int(round(self.lx / self.dx)) + 1

    @property
    def ny(self) -> int:
        return int(round(self.ly / self.dx)) + 1

    @property
    def dt(self) -> float:
        return 0.35 * self.dx**2 / (4 * self.alpha)


def grid(cfg: Config):
    x = np.linspace(-cfg.lx / 2, cfg.lx / 2, cfg.nx)
    y = np.linspace(-cfg.ly / 2, cfg.ly / 2, cfg.ny)
    return x, y


def gaussian_kernel(cfg: Config, w_m: float, power_w: float) -> np.ndarray:
    """Absorbed-flux kernel (W/m^2) of one Gaussian spotlet, on the FD grid.

    w_m is the 1/e^2 radius. The kernel is normalized numerically so its
    integral equals absorptivity * power_w regardless of grid resolution.
    """
    half = int(np.ceil(3 * w_m / cfg.dx))
    ax = np.arange(-half, half + 1) * cfg.dx
    kx, ky = np.meshgrid(ax, ax)
    kern = np.exp(-2 * (kx**2 + ky**2) / w_m**2)
    kern *= cfg.absorptivity * power_w / (kern.sum() * cfg.dx**2)
    return kern


def run_fd(cfg: Config, kernel: np.ndarray):
    """Transient FD with the given absorbed-flux kernel following the scan.

    Returns (final temperature field, laser column index, laser row index).
    The kernel is pasted at the nearest grid node of the laser position
    each step (error <= dx/2 = 2.5 um, negligible vs. 30 um spotlets).
    """
    x, y = grid(cfg)
    nx, ny, dxg, dt, alpha = cfg.nx, cfg.ny, cfg.dx, cfg.dt, cfg.alpha
    temp = np.full((ny, nx), cfg.ambient_c)
    src_per_step = kernel * (dt / (cfg.rho * cfg.cp * cfg.thickness_m))
    kh = kernel.shape[0] // 2
    iy0 = ny // 2
    steps = int(cfg.total_t / dt)

    lap = np.zeros_like(temp)
    for step in range(steps + 1):
        t = step * dt
        laser_x = cfg.x_start + cfg.speed_m_s * t
        ix0 = int(round((laser_x - x[0]) / dxg))

        lap[1:-1, 1:-1] = (
            temp[1:-1, 2:] - 2 * temp[1:-1, 1:-1] + temp[1:-1, :-2]
            + temp[2:, 1:-1] - 2 * temp[1:-1, 1:-1] + temp[:-2, 1:-1]
        ) / dxg**2
        temp[1:-1, 1:-1] += dt * alpha * lap[1:-1, 1:-1]

        # Clipped paste: wide kernels (side-lobe satellites) may overhang
        # the domain near the scan start; deposit the in-domain part.
        ylo, yhi = iy0 - kh, iy0 + kh + 1
        xlo, xhi = ix0 - kh, ix0 + kh + 1
        temp[max(ylo, 0):min(yhi, ny), max(xlo, 0):min(xhi, nx)] += \
            src_per_step[max(-ylo, 0):src_per_step.shape[0] - max(yhi - ny, 0),
                         max(-xlo, 0):src_per_step.shape[1] - max(xhi - nx, 0)]
        # Fixed-temperature edges (as in the original script)
        temp[0, :] = temp[-1, :] = temp[:, 0] = temp[:, -1] = cfg.ambient_c

    laser_x_final = cfg.x_start + cfg.speed_m_s * steps * dt
    ix_laser = int(round((laser_x_final - x[0]) / dxg))
    return temp, ix_laser, iy0


def _cache_key(cfg: Config, w_spot_m: float) -> Path:
    payload = json.dumps({**asdict(cfg), "w_spot": w_spot_m}, sort_keys=True)
    return CACHE_DIR / f"greens_{hashlib.md5(payload.encode()).hexdigest()[:12]}.npz"


def greens_field(cfg: Config, w_spot_m: float) -> dict:
    """Quasi-steady excess-temperature field (K per laser watt) of one
    spotlet of 1/e^2 radius w_spot_m moving at cfg.speed_m_s. Cached."""
    path = _cache_key(cfg, w_spot_m)
    if path.exists():
        d = np.load(path)
        return {"E": d["E"], "ix": int(d["ix"]), "iy": int(d["iy"])}
    kernel = gaussian_kernel(cfg, w_spot_m, power_w=1.0)
    temp, ix, iy = run_fd(cfg, kernel)
    E = temp - cfg.ambient_c  # excess T per 1 W of laser power
    np.savez_compressed(path, E=E, ix=ix, iy=iy)
    return {"E": E, "ix": ix, "iy": iy}


def _int_shift(a: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """Shift a 2D array by integer cells, filling with zeros (excess T ~ 0
    away from the track; shifts are small vs. the domain)."""
    out = np.zeros_like(a)
    ys = slice(max(oy, 0), a.shape[0] + min(oy, 0))
    xs = slice(max(ox, 0), a.shape[1] + min(ox, 0))
    yd = slice(max(-oy, 0), a.shape[0] + min(-oy, 0))
    xd = slice(max(-ox, 0), a.shape[1] + min(-ox, 0))
    out[ys, xs] = a[yd, xd]
    return out


def compose(green: dict, fracs: np.ndarray, offsets_m: np.ndarray, cfg: Config) -> np.ndarray:
    """Temperature field of a spot pattern by superposition.

    fracs: (K,) nonnegative power fractions (will be normalized to sum 1).
    offsets_m: (K, 2) spot offsets (dx_m, dy_m) relative to pattern center.
    """
    fracs = np.asarray(fracs, dtype=float)
    total = fracs.sum()
    if total <= 0:
        raise ValueError("at least one spot must have positive amplitude")
    fracs = fracs / total
    E = green["E"]
    field = np.zeros_like(E)
    for f, (ox_m, oy_m) in zip(fracs, offsets_m):
        if f < 1e-12:
            continue
        ox = int(round(ox_m / cfg.dx))
        oy = int(round(oy_m / cfg.dx))
        field += (f * cfg.power_w) * _int_shift(E, oy, ox)
    return cfg.ambient_c + field


METRIC_NAMES = [
    "peak_T_C",          # keyhole propensity proxy (vs. boil_c)
    "melt_width_um",     # transverse extent of T >= melt_c
    "melt_length_um",    # longitudinal extent
    "melt_area_um2",
    "G_rear_K_per_mm",   # mean |grad T| over the trailing solidification front
    "cool_rate_K_per_ms",  # v * mean |dT/dx| over that front (quasi-steady)
]


def metrics(field: np.ndarray, cfg: Config) -> np.ndarray:
    """Extract the 6 scalar responses used as ML labels. Returns zeros for
    the melt metrics when nothing reaches melt_c.

    G and the cooling rate are averaged over all melt-boundary cells in
    the trailing half of the pool (x behind the pool centroid), i.e. the
    solidification front. Averaging over the front instead of a single
    rearmost cell removes grid-threshold noise from the ML labels.
    """
    peak = float(field.max())
    mask = field >= cfg.melt_c
    if not mask.any():
        return np.array([peak, 0.0, 0.0, 0.0, 0.0, 0.0])
    ys, xs = np.nonzero(mask)
    width = (ys.max() - ys.min() + 1) * cfg.dx * 1e6
    length = (xs.max() - xs.min() + 1) * cfg.dx * 1e6
    area = mask.sum() * cfg.dx**2 * 1e12

    # Boundary cells: melted with at least one unmelted 4-neighbor.
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 2:] & mask[1:-1, :-2]
                            & mask[2:, 1:-1] & mask[:-2, 1:-1])
    boundary = mask & ~interior
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = False
    by, bx = np.nonzero(boundary)
    rear = bx <= xs.mean()  # trailing half = solidification front
    if not rear.any():
        rear = np.ones_like(bx, dtype=bool)
    by, bx = by[rear], bx[rear]
    dTdx = (field[by, bx + 1] - field[by, bx - 1]) / (2 * cfg.dx)
    dTdy = (field[by + 1, bx] - field[by - 1, bx]) / (2 * cfg.dx)
    G = float(np.hypot(dTdx, dTdy).mean())              # K/m
    cool = cfg.speed_m_s * float(np.abs(dTdx).mean())   # K/s
    return np.array([peak, width, length, area, G * 1e-3, cool * 1e-3])
