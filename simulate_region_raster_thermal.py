from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
OUT_DIR = ROOT / "thermal_region_raster_300W_400mms_25overlap"
OUT_DIR.mkdir(exist_ok=True)

OUT_GIF = OUT_DIR / "thermal_region_raster_25overlap.gif"
OUT_SUMMARY = OUT_DIR / "thermal_region_raster_25overlap_summary.png"
OUT_CSV = OUT_DIR / "thermal_region_raster_25overlap_history.csv"
OUT_SCRIPT_NOTE = OUT_DIR / "assumptions.txt"


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def cmap_temperature(temp: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    stops = np.array(
        [
            [18, 24, 38],
            [30, 64, 175],
            [8, 145, 178],
            [35, 180, 110],
            [245, 190, 70],
            [230, 80, 45],
            [255, 245, 210],
        ],
        dtype=float,
    )
    v = np.clip((temp - tmin) / max(1e-9, (tmax - tmin)), 0, 1)
    s = v * (len(stops) - 1)
    lo = np.floor(s).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    f = s - lo
    return (stops[lo] * (1 - f[..., None]) + stops[hi] * f[..., None]).astype(np.uint8)


def build_path(x_min, x_max, y_min, y_max, hatch, speed, dt):
    tracks = []
    ys = np.arange(y_min, y_max + 0.5 * hatch, hatch)
    t = 0.0
    points = []
    for i, y in enumerate(ys):
        xs = (x_min, x_max) if i % 2 == 0 else (x_max, x_min)
        length = abs(xs[1] - xs[0])
        duration = length / speed
        n = max(2, int(np.ceil(duration / dt)))
        for j in range(n):
            frac = j / (n - 1)
            x = xs[0] + frac * (xs[1] - xs[0])
            points.append((t, x, y, True, i + 1))
            t += dt
        tracks.append((xs[0], xs[1], y))
        # Short turn-around dwell with laser off.
        if i != len(ys) - 1:
            dwell = 0.20e-3
            n_dwell = max(1, int(np.ceil(dwell / dt)))
            for _ in range(n_dwell):
                points.append((t, xs[1], y, False, i + 1))
                t += dt
    return points, tracks


def add_colorbar(draw, x0, y0, h, tmin, tmax, fnt):
    for i in range(h):
        frac = 1 - i / max(1, h - 1)
        color = tuple(cmap_temperature(np.array([[tmin + frac * (tmax - tmin)]]), tmin, tmax)[0, 0])
        draw.line((x0, y0 + i, x0 + 24, y0 + i), fill=color)
    draw.rectangle((x0, y0, x0 + 24, y0 + h), outline="#172033", width=1)
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = y0 + h - frac * h
        draw.line((x0 + 24, y, x0 + 30, y), fill="#172033")
        draw.text((x0 + 36, y - 8), f"{tmin + frac * (tmax - tmin):.0f}", fill="#172033", font=fnt)


def main() -> None:
    power_w = 300.0
    speed_m_s = 0.400
    overlap = 0.25

    # Illustrative 316L-like surface conduction model.
    absorptivity = 0.35
    rho = 7900.0
    cp = 500.0
    k = 16.0
    alpha = k / (rho * cp)
    ambient_c = 25.0
    melt_c = 1375.0
    thickness_m = 0.20e-3
    beam_radius_m = 60e-6  # 1/e^2 radius
    beam_diameter_m = 2 * beam_radius_m
    hatch_m = beam_diameter_m * (1 - overlap)

    # Region to scan.
    region_x = 0.90e-3
    region_y = 0.45e-3
    x_min, x_max = -region_x / 2, region_x / 2
    y_min, y_max = -region_y / 2, region_y / 2

    margin = 0.35e-3
    lx = region_x + 2 * margin
    ly = region_y + 2 * margin
    nx, ny = 181, 131
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    dt = 0.36 * min(dx, dy) ** 2 / (4 * alpha)

    x = np.linspace(-lx / 2, lx / 2, nx)
    y = np.linspace(-ly / 2, ly / 2, ny)
    xx, yy = np.meshgrid(x, y)
    temp = np.full((ny, nx), ambient_c)

    path, tracks = build_path(x_min, x_max, y_min, y_max, hatch_m, speed_m_s, dt)
    eta_p = absorptivity * power_w

    def laplacian(t):
        p = np.pad(t, 1, mode="edge")
        return (
            (p[1:-1, 2:] - 2 * t + p[1:-1, :-2]) / dx**2
            + (p[2:, 1:-1] - 2 * t + p[:-2, 1:-1]) / dy**2
        )

    title = font(28)
    fnt = font(18)
    small = font(14)

    frames = []
    snapshots = []
    history = []
    max_seen = ambient_c
    save_every = max(1, len(path) // 100)
    snapshot_at = [0.20, 0.45, 0.70, 1.00]
    next_snap = 0

    def render(arr, laser_x, laser_y, t_s, track, laser_on, scale_max):
        rgb = cmap_temperature(arr, ambient_c, scale_max)
        img = Image.fromarray(rgb, mode="RGB").resize((760, 560), Image.Resampling.BICUBIC)
        draw = ImageDraw.Draw(img)
        # Scan region border.
        rx0 = int((x_min - x[0]) / (x[-1] - x[0]) * img.width)
        rx1 = int((x_max - x[0]) / (x[-1] - x[0]) * img.width)
        ry0 = int((y_min - y[0]) / (y[-1] - y[0]) * img.height)
        ry1 = int((y_max - y[0]) / (y[-1] - y[0]) * img.height)
        draw.rectangle((rx0, ry0, rx1, ry1), outline="white", width=2)
        px = int((laser_x - x[0]) / (x[-1] - x[0]) * img.width)
        py = int((laser_y - y[0]) / (y[-1] - y[0]) * img.height)
        color = "#FFFFFF" if laser_on else "#CBD5E1"
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=color, width=2)
        draw.text((14, 12), f"t = {t_s*1e3:.2f} ms", fill="white", font=fnt)
        draw.text((14, 40), f"track {track}/{len(tracks)} | Tmax {arr.max():.0f} deg C", fill="white", font=fnt)
        return img

    for idx, (t, laser_x, laser_y, laser_on, track) in enumerate(path):
        if laser_on:
            q_surface = (2 * eta_p / (np.pi * beam_radius_m**2)) * np.exp(
                -2 * ((xx - laser_x) ** 2 + (yy - laser_y) ** 2) / beam_radius_m**2
            )
            source = q_surface / (rho * cp * thickness_m)
        else:
            source = 0.0
        temp += dt * (alpha * laplacian(temp) + source)
        temp[0, :] = ambient_c
        temp[-1, :] = ambient_c
        temp[:, 0] = ambient_c
        temp[:, -1] = ambient_c

        max_seen = max(max_seen, float(temp.max()))
        melt_area = float((temp >= melt_c).sum() * dx * dy * 1e6)
        history.append((t * 1e3, track, laser_x * 1e3, laser_y * 1e3, float(temp.max()), melt_area))

        frac = idx / max(1, len(path) - 1)
        if next_snap < len(snapshot_at) and frac >= snapshot_at[next_snap]:
            snapshots.append((t, laser_x, laser_y, track, temp.copy()))
            next_snap += 1
        if idx % save_every == 0 or idx == len(path) - 1:
            frames.append(render(temp.copy(), laser_x, laser_y, t, track, laser_on, max(max_seen, melt_c + 50)))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "track", "laser_x_mm", "laser_y_mm", "peak_temperature_C", "area_above_1375C_mm2"])
        writer.writerows((f"{a:.6f}", b, f"{c:.6f}", f"{d:.6f}", f"{e:.3f}", f"{g:.8f}") for a, b, c, d, e, g in history)

    if frames:
        frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=60, loop=0)

    # Summary sheet
    summary = Image.new("RGB", (1500, 950), "#F7F8FA")
    draw = ImageDraw.Draw(summary)
    draw.text((54, 38), "Raster scan thermal evolution with 25% overlap", fill="#172033", font=title)
    draw.text((56, 84), "300 W, 400 mm/s; adjacent hatch spacing = 75% of beam diameter", fill="#64748B", font=fnt)
    scale_max = max(max_seen, melt_c + 50)
    positions = [(70, 150), (520, 150), (70, 520), (520, 520)]
    for snap, pos in zip(snapshots, positions):
        t, lx0, ly0, track, arr = snap
        img = render(arr, lx0, ly0, t, track, True, scale_max).resize((390, 288), Image.Resampling.BICUBIC)
        summary.paste(img, pos)
        draw.rectangle((pos[0], pos[1], pos[0] + 390, pos[1] + 288), outline="#CBD5E1", width=2)
        draw.text((pos[0], pos[1] - 28), f"{t*1e3:.1f} ms | track {track}", fill="#172033", font=fnt)

    add_colorbar(draw, 965, 155, 350, ambient_c, scale_max, small)
    draw.text((960, 124), "temperature (deg C)", fill="#172033", font=small)

    hist = np.array(history, dtype=float)
    px0, py0, px1, py1 = 1045, 610, 1440, 840
    draw.rectangle((px0, py0, px1, py1), fill="white", outline="#CBD5E1", width=2)
    tt, peak = hist[:, 0], hist[:, 4]
    xs = px0 + 42 + (tt - tt.min()) / (tt.max() - tt.min()) * (px1 - px0 - 64)
    ys = py1 - 34 - (peak - ambient_c) / (peak.max() - ambient_c) * (py1 - py0 - 62)
    draw.line(list(map(tuple, np.column_stack([xs, ys]))), fill="#E11D48", width=4)
    draw.text((px0 + 30, py0 + 18), "Peak temperature", fill="#172033", font=fnt)
    draw.text((px0 + 30, py0 + 48), f"max {peak.max():.0f} deg C", fill="#E11D48", font=fnt)
    draw.text((px0 + 30, py1 - 26), "time (ms)", fill="#64748B", font=small)

    draw.rounded_rectangle((1045, 155, 1440, 505), radius=14, fill="#172033")
    notes = [
        "Scan assumptions",
        f"Region: {region_x*1e3:.2f} x {region_y*1e3:.2f} mm",
        f"Beam diameter: {beam_diameter_m*1e6:.0f} micrometers",
        f"Hatch: {hatch_m*1e6:.0f} micrometers",
        f"Tracks: {len(tracks)} serpentine",
        f"Absorptivity: {absorptivity:.2f}",
        "No latent heat, vaporization, radiation, convection, or powder packing",
    ]
    ytxt = 184
    for i, txt in enumerate(notes):
        draw.text((1070, ytxt), txt, fill="#F2B84B" if i == 0 else "#FFFFFF", font=fnt if i == 0 else small)
        ytxt += 42 if i == 0 else 31
    summary.save(OUT_SUMMARY)

    OUT_SCRIPT_NOTE.write_text(
        "\n".join(
            [
                "Raster scan thermal simulation assumptions",
                f"Power: {power_w} W",
                f"Speed: {speed_m_s*1000:.0f} mm/s",
                f"Overlap: {overlap*100:.0f}%",
                f"Beam radius: {beam_radius_m*1e6:.0f} micrometers, 1/e^2",
                f"Hatch spacing: {hatch_m*1e6:.1f} micrometers",
                f"Region: {region_x*1e3:.3f} mm x {region_y*1e3:.3f} mm",
                "Material: 316L-like constants, rho=7900 kg/m3, cp=500 J/kg-K, k=16 W/m-K",
                "Model: simplified 2D conduction with effective 200 micrometer layer thickness.",
            ]
        ),
        encoding="utf-8",
    )

    print(OUT_SUMMARY)
    print(OUT_GIF)
    print(OUT_CSV)
    print(f"tracks={len(tracks)} hatch_um={hatch_m*1e6:.1f} peak_C={max_seen:.1f}")


if __name__ == "__main__":
    main()
