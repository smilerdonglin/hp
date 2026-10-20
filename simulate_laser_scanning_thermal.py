from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_DIR = ROOT / "thermal_scan_300W_400mms"
OUT_DIR.mkdir(exist_ok=True)

OUT_SUMMARY = OUT_DIR / "thermal_scan_summary.png"
OUT_GIF = OUT_DIR / "thermal_scan_evolution.gif"
OUT_CSV = OUT_DIR / "thermal_scan_peak_history.csv"


def cmap_temperature(temp: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    stops = np.array(
        [
            [20, 24, 38],
            [30, 64, 175],
            [8, 145, 178],
            [35, 180, 110],
            [245, 190, 70],
            [230, 80, 45],
            [255, 245, 210],
        ],
        dtype=float,
    )
    v = np.clip((temp - tmin) / (tmax - tmin), 0, 1)
    s = v * (len(stops) - 1)
    lo = np.floor(s).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    f = s - lo
    return (stops[lo] * (1 - f[..., None]) + stops[hi] * f[..., None]).astype(np.uint8)


def text_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_colorbar(draw: ImageDraw.ImageDraw, x0, y0, h, tmin, tmax, font):
    for i in range(h):
        v = 1 - i / max(1, h - 1)
        color = tuple(cmap_temperature(np.array([[tmin + v * (tmax - tmin)]]), tmin, tmax)[0, 0])
        draw.line((x0, y0 + i, x0 + 24, y0 + i), fill=color, width=1)
    draw.rectangle((x0, y0, x0 + 24, y0 + h), outline="#172033", width=1)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = y0 + h - frac * h
        t = tmin + frac * (tmax - tmin)
        draw.line((x0 + 24, y, x0 + 30, y), fill="#172033", width=1)
        draw.text((x0 + 36, y - 8), f"{t:.0f}", fill="#172033", font=font)
    draw.text((x0 - 8, y0 - 28), "deg C", fill="#172033", font=font)


def main() -> None:
    # User inputs
    power_w = 300.0
    speed_m_s = 0.400

    # Assumed material: 316L stainless steel, effective 2D surface/track model.
    absorptivity = 0.35
    rho = 7900.0
    cp = 500.0
    k = 16.0
    alpha = k / (rho * cp)
    thickness_m = 0.20e-3
    ambient_c = 25.0
    melt_c = 1375.0

    # Gaussian laser radius at 1/e^2 intensity.
    beam_radius_m = 60e-6

    # Domain and explicit finite-difference settings.
    lx = 2.4e-3
    ly = 1.2e-3
    nx = 241
    ny = 121
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    dt = 0.35 * min(dx, dy) ** 2 / (4 * alpha)
    total_t = 4.0e-3
    steps = int(total_t / dt)
    save_every = max(1, steps // 80)

    x = np.linspace(-lx / 2, lx / 2, nx)
    y = np.linspace(-ly / 2, ly / 2, ny)
    xx, yy = np.meshgrid(x, y)
    temp = np.full((ny, nx), ambient_c, dtype=float)

    # Laser starts before left edge and traverses toward right.
    x_start = -0.85e-3
    y_laser = 0.0
    eta_p = absorptivity * power_w
    denom = np.pi * beam_radius_m**2 / 2

    frames = []
    snapshots = []
    history = []

    title_font = text_font(30)
    font = text_font(18)
    small = text_font(14)

    def laplacian(t):
        padded = np.pad(t, 1, mode="edge")
        return (
            (padded[1:-1, 2:] - 2 * t + padded[1:-1, :-2]) / dx**2
            + (padded[2:, 1:-1] - 2 * t + padded[:-2, 1:-1]) / dy**2
        )

    def render_frame(t_s: float, laser_x: float, arr: np.ndarray, max_for_scale: float):
        rgb = cmap_temperature(arr, ambient_c, max_for_scale)
        img = Image.fromarray(rgb, mode="RGB").resize((720, 360), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(img)
        px = int((laser_x - x[0]) / (x[-1] - x[0]) * img.width)
        py = img.height // 2
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline="white", width=2)
        draw.text((14, 12), f"t = {t_s*1e3:.2f} ms", fill="white", font=font)
        draw.text((14, 38), f"Tmax = {arr.max():.0f} deg C", fill="white", font=font)
        return img

    max_seen = ambient_c
    for step in range(steps + 1):
        t = step * dt
        laser_x = x_start + speed_m_s * t
        q_surface = (2 * eta_p / (np.pi * beam_radius_m**2)) * np.exp(
            -2 * ((xx - laser_x) ** 2 + (yy - y_laser) ** 2) / beam_radius_m**2
        )
        source = q_surface / (rho * cp * thickness_m)
        temp += dt * (alpha * laplacian(temp) + source)

        # Simple edge clamping to ambient avoids artificial wraparound heat.
        temp[0, :] = ambient_c
        temp[-1, :] = ambient_c
        temp[:, 0] = ambient_c
        temp[:, -1] = ambient_c

        max_seen = max(max_seen, float(temp.max()))
        melt_area = float((temp >= melt_c).sum() * dx * dy * 1e6)
        history.append((t * 1e3, laser_x * 1e3, float(temp.max()), melt_area))

        if step % save_every == 0 or step == steps:
            frames.append(render_frame(t, laser_x, temp.copy(), max(ambient_c + 10, max_seen)))
        if len(snapshots) < 4 and t >= [0.8e-3, 1.6e-3, 2.4e-3, 3.2e-3][len(snapshots)]:
            snapshots.append((t, laser_x, temp.copy()))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "laser_x_mm", "peak_temperature_C", "area_above_1375C_mm2"])
        writer.writerows((f"{a:.6f}", f"{b:.6f}", f"{c:.3f}", f"{d:.8f}") for a, b, c, d in history)

    # Summary panel
    summary = Image.new("RGB", (1500, 950), "#F7F8FA")
    draw = ImageDraw.Draw(summary)
    draw.text((54, 38), "Thermal evolution during laser scanning", fill="#172033", font=title_font)
    draw.text((56, 84), "300 W laser, 400 mm/s scan speed; simplified 2D conductive surface model", fill="#64748B", font=font)

    global_max = max(float(s[2].max()) for s in snapshots) if snapshots else max_seen
    scale_max = max(global_max, melt_c + 100)
    positions = [(70, 155), (540, 155), (70, 545), (540, 545)]
    for (t, laser_x, arr), pos in zip(snapshots, positions):
        img = render_frame(t, laser_x, arr, scale_max).resize((420, 210), Image.Resampling.BICUBIC)
        summary.paste(img, pos)
        draw.rectangle((pos[0], pos[1], pos[0] + 420, pos[1] + 210), outline="#CBD5E1", width=2)
        draw.text((pos[0], pos[1] - 28), f"{t*1e3:.1f} ms", fill="#172033", font=font)

    add_colorbar(draw, 1005, 170, 360, ambient_c, scale_max, small)

    hist = np.array(history)
    px0, py0, px1, py1 = 1080, 610, 1440, 840
    draw.rectangle((px0, py0, px1, py1), fill="white", outline="#CBD5E1", width=2)
    tt = hist[:, 0]
    pp = hist[:, 2]
    xs = px0 + 42 + (tt - tt.min()) / (tt.max() - tt.min()) * (px1 - px0 - 64)
    ys = py1 - 35 - (pp - ambient_c) / (pp.max() - ambient_c) * (py1 - py0 - 60)
    draw.line(list(map(tuple, np.column_stack([xs, ys]))), fill="#E11D48", width=4)
    draw.text((px0 + 42, py0 + 18), "Peak temperature history", fill="#172033", font=font)
    draw.text((px0 + 42, py1 - 28), "time (ms)", fill="#64748B", font=small)
    draw.text((px0 + 42, py0 + 45), f"max = {pp.max():.0f} deg C", fill="#E11D48", font=font)

    assumptions = [
        "Assumptions",
        "Material: 316L stainless steel",
        f"Absorptivity: {absorptivity:.2f}",
        f"Beam radius: {beam_radius_m*1e6:.0f} micrometers",
        f"Effective layer thickness: {thickness_m*1e6:.0f} micrometers",
        "No convection, radiation, vaporization, latent heat, or powder effects",
    ]
    draw.rounded_rectangle((1080, 170, 1440, 530), radius=12, fill="#172033")
    ytxt = 198
    for i, txt in enumerate(assumptions):
        draw.text((1105, ytxt), txt, fill="#F2B84B" if i == 0 else "#FFFFFF", font=font if i == 0 else small)
        ytxt += 44 if i == 0 else 32

    summary.save(OUT_SUMMARY)
    if frames:
        frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=65, loop=0)

    print(OUT_SUMMARY)
    print(OUT_GIF)
    print(OUT_CSV)
    print(f"peak_C={max_seen:.1f}")


if __name__ == "__main__":
    main()
