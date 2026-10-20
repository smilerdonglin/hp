# Beam Shaping (code-only repository)

This repository holds the scripts, data, and notes for the Civan CBC/OPA beam-shaping
LPBF project led by Dong Lin (Oregon State University) with Auburn collaborators.
The large PowerPoint, PDF, and Word files live in the Google Drive folder
`G:\My Drive\Faculty\Research Projects\3D Printing of Metal\Beam Shaping` and are
deliberately not in git (see .gitignore). Cloud sessions see only what is here.

## People

- Dong Lin: PI, Oregon State (dong.lin@oregonstate.edu, GitHub smilerdonglin)
- Ryan Bertelsen: PhD student, co-corresponding author on the manuscript
- Devin Roach: Oregon State co-corresponding author
- Auburn co-corresponding author: rzw0090@auburn.edu

## Project context

- Manuscript in progress: single tracks on 316L with five beam shapes (Gaussian
  reference plus CBC "Smile", "Arrow", "A", "Turtle"), two powers, fixed speed.
  In-situ IR thermography and high-speed imaging, ex-situ cross-sections.
  Key results: CBC shapes reduce thermal gradients and cooling rates, change
  conduction/keyhole regime by shape alone, Smile beam gives about 13% spatter reduction.
- Closest prior work: Bayat et al. 2024, Addit. Manuf. 93:104420 (simulation only,
  axisymmetric core:ring, no Civan hardware). Our differentiation: real DBL hardware,
  arbitrary asymmetric shapes, 316L, ML inverse design.
- DBL hardware truth (Weber et al. JMMP 9:85, 2025): a shape is a time-sequenced single
  main lobe (about 60% of power), dwell-weighted, about 6% side lobes, 68 um spot over a
  ~260 um field at f = 0.75 m. Melt pool acts as a ~2 ms low-pass filter (Sawannia 2024).
- Regime classifier: peak-to-mean ratio Phi with H_max/h_s thresholds 6.25 (conduction to
  transition) and 8.65 (transition to keyhole), Grunewald AIME 2025.

## Layout

- `simulate_beam_energy.py`: Gaussian vs ring beam energy distribution.
- `simulate_laser_scanning_thermal.py`, `simulate_region_raster_thermal.py`: 2D explicit
  finite-difference thermal models, 316L properties. Outputs in
  `thermal_scan_300W_400mms/` and `thermal_region_raster_300W_400mms_25overlap/`.
- `ML Beam Design/`: inverse-design pipeline. Read its README.md first. Run order:
  `generate_dataset.py 3000`, `train_surrogate.py`, `design_beam.py`.
  The `cache/` Green's-function .npz files are regenerated on first run and are not in git.
- `Papers/Literature Notes - Beam Shaping.md`: synthesis of 22 papers (2026-07-11).

## Conventions

- Python 3.12, numpy, scipy, matplotlib, Pillow. No other runtime dependencies.
- Scripts write outputs next to themselves using `Path.cwd()`, so run them from their
  own folder.
- Keep large binaries out of git. Figures under 1 MB are fine.
- Known caveat: the 2nd-moment Phi convention over-penalizes sparse multi-lobe patterns.
  Treat peak temperature as the binding keyhole indicator.
