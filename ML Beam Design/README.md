# ML Beam Design — inverse design of CBC beam shapes

Machine-learning pipeline that *designs* a Civan CBC/OPA beam shape to hit a
melt-pool target, instead of testing hand-picked shapes. Companion to the
manuscript in `../AMET Experiments/Civan laser draft V5.2 Ryan.docx`.

## Idea

1. **Parameterization** — a beam shape is a 7×7 grid of spotlets (50 µm pitch,
   30 µm 1/e² radius each) with programmable power fractions: a 49-dim design
   vector. This mirrors how the OPA renders shapes and can be re-gridded to
   match the real DBL 6 shape-file format.
2. **Fast physics** — the 2D thin-plate FD model (refactored from
   `../simulate_laser_scanning_thermal.py`) is *linear*, so the quasi-steady
   field of any pattern is a superposition of one precomputed single-spotlet
   response. One FD run (~15 s, cached) buys a ~2 ms forward model that is
   **exact** (validated at 0.000% error vs direct FD).
3. **Surrogate** — a small MLP (numpy-only, `surrogate.py`) maps the 49
   fractions → 6 thermal metrics. It exists so the pipeline carries over
   unchanged when the linear model is replaced by a high-fidelity simulator
   (keyhole/CFD) or by experimental data, and to provide gradients.
4. **Inverse design** — gradient descent *through* the surrogate (softmax over
   spot logits, multi-start), then a local search polish on the exact forward
   model, then full transient FD verification of the winner.

## Files

| file | role |
|---|---|
| `thermal_model.py` | Config (316L, 300 W, 400 mm/s), FD solver for arbitrary kernels, Green's-function superposition, metric extraction |
| `beam_shapes.py` | spot-grid parameterization, presets (ring, line, arcs), random shape families for dataset coverage |
| `generate_dataset.py` | Green's functions + FD validation + N-sample dataset (`outputs/dataset.npz/.csv`) |
| `surrogate.py` | dependency-free MLP with Adam and input gradients |
| `train_surrogate.py` | training + validation R² + parity plot |
| `design_beam.py` | 3-stage inverse design + baselines + comparison figure/CSV |

Run in order (Python 3.12, needs numpy/scipy/matplotlib):

```
python generate_dataset.py 3000
python train_surrogate.py
python design_beam.py
```

## Results (2026-07-11, hardware-realistic rendering)

The renderer now includes the Weber et al. 2025 hardware physics: 40 µm
spotlets (grid exactly spans the placement field), four ~6%-peak side-lobe
satellites per position, field-envelope droop (edge cells deliver ~53%),
and the Φ / H_max/h_s regime metric (Grünewald AIME 2025) as an 8th learned
output and keyhole constraint. Superposition still matches direct FD to
≤0.2%. Cross-check: the 60 µm Gaussian reference gets Φ = 2.00 and
H_max/h_s = 8.50 — right at the literature keyhole threshold (8.65),
independently consistent with our thermal peak (2880 °C vs 2900 °C boiling).

Objective: minimize solidification-front gradient G subject to melt width ≥
Gaussian reference, 1800 °C ≤ peak T ≤ boiling (the superheat floor guards
against the optimizer "gaming" G by barely melting), H_max/h_s ≤ 8.65.

FD-verified designed beam vs 60 µm Gaussian at 300 W, 400 mm/s:
**G −43%, melt width −4%, peak 1797 °C, cooling −5%**, using 5 active spots
(a trailing-weighted asymmetric cluster). The design sits exactly on the
superheat and width constraints — the realistic optics (satellite losses +
envelope droop) genuinely shrink what is achievable vs the idealized model
(which reached G −58%). H_max/h_s = 8.94 slightly exceeds the 8.65
threshold: the 2nd-moment Φ convention over-penalizes sparse multi-lobe
patterns (satellites inflate the 2nd-moment radius), while the thermal
model shows a comfortably conduction-mode pool; treat H_max/h_s as
conservative for CBC patterns and peak-T as the binding keyhole indicator.

Surrogate validation R² (8 outputs): peak 0.96, width 0.93, length 0.96,
area 0.97, G 0.85, cooling 0.50 (report-only), Φ 0.89, H_max/h_s 0.96.

Note: the physics is y-symmetric, so optima come in mirror pairs and the
returned design may be arbitrarily one-sided; add a symmetry constraint if
a centered track is required.

## Hardware-realism constraints from the literature (2026-07-11 survey)

Full details in `../Papers/Literature Notes - Beam Shaping.md`. Key corrections the
parameterization should adopt (from Weber et al., JMMP 9:85, 2025 and related):

- A DBL "shape" is a time-sequence of ONE moving main lobe (~60% of power); per-cell
  "power fraction" is physically a dwell-time fraction (integer repeat counts).
  Valid as a static intensity map in the quasi-static regime (µs refresh ≪ ~ms
  melt-pool response — Sawannia 2024 measured a ~2 ms capillary time constant).
- Unavoidable side lobes (~6% of peak, four-fold) and a field-envelope droop
  (edge spots lose intensity, down to 54% at the placement-field edge) should be
  added to the rendered intensity.
- Spot size is set by optics (≈91·f[m] µm at 1/e²; ~68 µm at f=0.75 m over a
  ~260 µm placement field) — verify our 30 µm/50 µm-pitch grid against the actual
  AMET machine optics before exporting designs.
- Useful physics-based feature/constraint: peak-to-mean intensity ratio Φ with
  H_max/h_s = (Φ/2)(H/h_s); conduction→transition at 6.25±0.85, transition→keyhole
  at 8.65±0.30 across all shapes (Grünewald AIME 2025).

## Limitations / next steps

- Linear conduction only: no keyhole, fluid flow, latent heat, vaporization,
  or powder. Peak T vs boiling is a keyhole-propensity *proxy*; spatter
  claims must remain experimental.
- Calibrate absorptivity / effective properties against the in-situ IR data
  before drawing quantitative conclusions.
- The dataset covers static shapes; the temporal dimension (kHz shape
  sequences) is a natural extension — add a sequence index to the design
  vector and reuse the same superposition trick per sub-shape.
- Metrics are y-symmetric physics, but the optimizer may return asymmetric
  optima; add a symmetry constraint if desired.
- Experimental closure: export the designed spot pattern to the Civan shape
  format, print single tracks, compare IR thermography vs prediction.
