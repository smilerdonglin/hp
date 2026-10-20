from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_PNG = ROOT / "gaussian_vs_ring_beam_energy_distribution.png"
OUT_CSV = ROOT / "gaussian_vs_ring_beam_radial_profile.csv"


def inferno_like(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0, 0, 4],
            [50, 16, 91],
            [120, 28, 109],
            [185, 55, 84],
            [230, 105, 45],
            [252, 185, 55],
            [252, 255, 164],
        ],
        dtype=float,
    )
    values = np.clip(values, 0, 1)
    scaled = values * (len(stops) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    t = scaled - lo
    rgb = stops[lo] * (1 - t[..., None]) + stops[hi] * t[..., None]
    return rgb.astype(np.uint8)


def render_heatmap(arr: np.ndarray, size: int = 440) -> Image.Image:
    norm = arr / arr.max()
    rgb = inferno_like(norm)
    img = Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.BICUBIC)
    return img


def draw_profile(draw: ImageDraw.ImageDraw, box, x_mm, g, ring, font):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="white", outline="#CBD5E1", width=2)
    pad_l, pad_r, pad_t, pad_b = 58, 22, 24, 46
    px0, px1 = x0 + pad_l, x1 - pad_r
    py0, py1 = y0 + pad_t, y1 - pad_b
    draw.line((px0, py1, px1, py1), fill="#172033", width=2)
    draw.line((px0, py0, px0, py1), fill="#172033", width=2)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        x = px0 + frac * (px1 - px0)
        draw.line((x, py1, x, py1 + 5), fill="#64748B", width=1)
        label = f"{x_mm[0] + frac * (x_mm[-1] - x_mm[0]):.1f}"
        draw.text((x - 12, py1 + 9), label, fill="#64748B", font=font)
    for frac in [0, 0.5, 1]:
        y = py1 - frac * (py1 - py0)
        draw.line((px0 - 5, y, px0, y), fill="#64748B", width=1)
        draw.text((x0 + 12, y - 7), f"{frac:.1f}", fill="#64748B", font=font)

    def to_points(vals):
        xs = px0 + (x_mm - x_mm[0]) / (x_mm[-1] - x_mm[0]) * (px1 - px0)
        ys = py1 - vals / max(g.max(), ring.max()) * (py1 - py0)
        return list(map(tuple, np.column_stack([xs, ys]).astype(float)))

    draw.line(to_points(g), fill="#E11D48", width=4, joint="curve")
    draw.line(to_points(ring), fill="#0E6F6A", width=4, joint="curve")
    draw.text((x0 + 70, y0 + 16), "centerline normalized intensity", fill="#172033", font=font)
    draw.text((x1 - 245, y0 + 16), "Gaussian", fill="#E11D48", font=font)
    draw.text((x1 - 132, y0 + 16), "Ring", fill="#0E6F6A", font=font)
    draw.text((px0 + (px1 - px0) / 2 - 38, y1 - 26), "x (mm)", fill="#64748B", font=font)


def main() -> None:
    n = 501
    extent_mm = 2.5
    x = np.linspace(-extent_mm, extent_mm, n)
    y = np.linspace(-extent_mm, extent_mm, n)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    # Typical illustrative values, not fitted to a specific source:
    # w0 is the 1/e^2 Gaussian radius; r0 is ring radius; wr controls ring thickness.
    w0_mm = 0.60
    r0_mm = 0.65
    wr_mm = 0.24

    gaussian = np.exp(-2 * rr**2 / w0_mm**2)
    ring = np.exp(-2 * (rr - r0_mm) ** 2 / wr_mm**2)

    dx = x[1] - x[0]
    gaussian /= gaussian.sum() * dx * dx
    ring /= ring.sum() * dx * dx

    g_norm = gaussian / gaussian.max()
    r_norm = ring / ring.max()

    center = n // 2
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "gaussian_normalized", "ring_normalized"])
        for xi, gi, ri in zip(x, g_norm[center, :], r_norm[center, :]):
            writer.writerow([f"{xi:.5f}", f"{gi:.8f}", f"{ri:.8f}"])

    canvas = Image.new("RGB", (1500, 950), "#F7F8FA")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        h_font = ImageFont.truetype("arial.ttf", 28)
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        title_font = h_font = font = small = ImageFont.load_default()

    draw.text((60, 42), "Simulated energy distribution: Gaussian vs ring beam", fill="#172033", font=title_font)
    draw.text((62, 96), "Both beams are normalized to the same integrated energy over the grid.", fill="#64748B", font=font)

    heat_size = 440
    g_img = render_heatmap(gaussian, heat_size)
    r_img = render_heatmap(ring, heat_size)
    canvas.paste(g_img, (80, 180))
    canvas.paste(r_img, (565, 180))
    draw.rectangle((80, 180, 80 + heat_size, 180 + heat_size), outline="#CBD5E1", width=2)
    draw.rectangle((565, 180, 565 + heat_size, 180 + heat_size), outline="#CBD5E1", width=2)
    draw.text((80, 140), "Gaussian beam", fill="#172033", font=h_font)
    draw.text((565, 140), "Ring beam", fill="#172033", font=h_font)
    draw.text((80, 632), "Central hot spot; intensity decays monotonically with radius.", fill="#243047", font=small)
    draw.text((565, 632), "Energy redistributed into annulus; center is deliberately cooler.", fill="#243047", font=small)

    # Draw radius markers.
    for x0 in [80, 565]:
        cx = x0 + heat_size // 2
        cy = 180 + heat_size // 2
        draw.line((cx, cy, cx + 110, cy), fill="white", width=3)
        draw.text((cx + 112, cy - 12), "r", fill="white", font=small)

    draw_profile(draw, (1040, 180, 1440, 620), x, g_norm[center, :], r_norm[center, :], small)

    notes = [
        f"Gaussian: I(r) = exp(-2r^2/w0^2), w0 = {w0_mm:.2f} mm",
        f"Ring: I(r) = exp(-2(r-r0)^2/wr^2), r0 = {r0_mm:.2f} mm, wr = {wr_mm:.2f} mm",
        "Interpretation: the ring beam lowers peak center energy and moves heat outward,",
        "which can reduce central overheating/keyhole tendency while widening the melt pool.",
    ]
    y0 = 720
    for i, note in enumerate(notes):
        draw.text((80, y0 + i * 38), note, fill="#172033" if i < 2 else "#243047", font=font)

    peak_ratio = ring.max() / gaussian.max()
    draw.rounded_rectangle((1040, 700, 1440, 860), radius=14, fill="#172033")
    draw.text((1070, 728), "Same total energy", fill="#FFFFFF", font=h_font)
    draw.text((1070, 775), f"Ring peak / Gaussian peak = {peak_ratio:.2f}", fill="#F2B84B", font=font)
    draw.text((1070, 812), "for this illustrative parameter set", fill="#CBD5E1", font=small)

    canvas.save(OUT_PNG)
    print(OUT_PNG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
