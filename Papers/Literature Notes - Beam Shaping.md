# Beam Shaping Literature Notes (22 papers)

Compiled 2026-07-11 from a full read of `Papers/Civan` (13), `Papers/nLight` (8), and
`Papers/Fundamental` (1). Companion to the manuscript in
`AMET Experiments/Civan laser draft V5.2 Ryan.docx` and the ML inverse-design
pipeline in `ML Beam Design/`.

---

## 1. Executive positioning summary

**Closest prior work — Bayat et al. 2024** (*Addit. Manuf.* 93, 104420; DTU/FAU/RWTH):
"Exploring spatial beam shaping in LPBF: high-fidelity simulation and in-situ
monitoring." Despite the title, it contains **no Civan experiments**: single tracks
used an nLight AFX1000 (Ti6Al4V), in-situ X-ray used a Coherent FL-ARM. The beam is
a **1-parameter analytical axisymmetric core:ring function** in FLOW-3D — it cannot
even express asymmetric shapes. Differentiation axes for our paper:
1. Real Civan DBL hardware vs idealized analytical beams.
2. Arbitrary asymmetric 2-D shapes (Smile/Arrow/A/Turtle) vs a single core:ring scalar.
3. 316L vs Ti6Al4V.
4. IR thermography + high-speed optical imaging (lab-deployable) vs synchrotron X-ray.
5. **ML inverse design (shape prescribed to hit a target) vs forward observation.**
6. Beam represented in the hardware's actuator space (spot/dwell grid) vs a continuum.

**Hardware-landscape argument for ML:** three dynamic beam-shaping technologies exist —
AOD time-multiplexed multi-spot (Grünewald 2022, µs refresh, small deflection range,
elliptical spots), nLight AFX in-source core/ring (**7 discrete presets**, ms switching),
and CBC/OPA (continuous, high-dimensional, ns–µs). Only CBC has a design space large
enough that hand-picking shapes is hopeless — that is the case for inverse design.
Grünewald 2022 states it directly: "depending on the targeted process or component
property, different temperature distributions and beam shapes are necessary."

**The nLight group concedes ring is not proven optimal** (Metals 2021: "the maximum
size of a stable melt pool has still to be investigated... productivity could
potentially be increased even further"; AML 2025: the decisive keyhole boundary is not
predictable by conduction physics but *is* controllable by intensity distribution).

---

## 2. What the Civan DBL actually is (Weber et al. 2025 — the primary hardware citation)

**Weber, Wagner, Peter, Hagenlocher (IFSW) + Spira, Urbach, Shekel, Vidne (Civan),
*J. Manuf. Mater. Process.* 9, 85 (2025)** (peer-reviewed twin of the SSRN preprint).

- DBL 6-14 kW: **6×6 array of fundamental-mode fibers** (M² ≈ 1.2 each), 1064 nm,
  collimated array Ø ≈ 18 mm. (OPA6 welding variant: 32 fibers, up to 14 kW.)
- One **"CBC point"** at a time: main lobe (~**60% of power**, near-diffraction-limited)
  + **side lobes (~6% of peak** at relative offset 0.36, 4-fold square-array symmetry).
  Side lobes are unavoidable; "dynamic spatial filtering" is only a future capability.
- A **shape = time-sequence of points** on a drawing grid (Shape Generator: 33×33 in
  IWS paper). **Cell weighting = integer dwell-repeat count N_C,i** (more dwell = more
  energy/area, NOT higher intensity). Max OPA modulation **80 MHz** → min dwell
  **12.5 ns**; **≤1024 points/shape; ≤14 shapes/sequence**.
- Geometry scales with focal length: d_MainLobe ≈ 91·f[m] µm (1/e²; 43 µm FWHM at
  f=0.75 m); point-placement half-field x_set,max ≈ 172·f[m] µm (≈1.9·d_MainLobe);
  active area half-width ≈ 473·f[m] µm. At f=0.75 m: **68 µm spots over a ~260 µm
  placement field**. Off-center points lose peak intensity (Gaussian envelope,
  d_Env ≈ 610·f[m] µm; down to 54% at field edge where a side lobe grows to equal it).
- Focus: DoF ±5.1·f² mm; electro-optic z-steering ±12.8·f² mm.
- Three process regimes vs process time constant t_Proc: **quasi-static**
  (t_SR ≪ t_Proc → melt sees time-averaged shape), dynamic (wobble), resonant.
- Total absorbed energy is roughly **shape-invariant** (~66–72% keyhole absorptance
  across patterns, Wagner PhotonicsViews 2022 ray-tracing on X-ray keyholes): shape
  redistributes *where* energy lands, not net coupling. (Consistent with Grünewald
  Metals 2021: equal line energy → equal molten area regardless of profile.)
- Weber et al. explicitly name **"online AI optimization"** of DBL processes as a
  future direction — direct license for our ML contribution.

**Consequences for the ML pipeline (`ML Beam Design/`):**
1. Power fractions ≙ dwell-time fractions (integer-quantized by repeat counts) —
   a continuous vector is a fine surrogate; say so explicitly.
2. Add a **side-lobe pedestal** (~6% four-fold satellites) to the rendered intensity.
3. Add the **field-envelope droop**: edge cells of a 7×7 @ 50 µm grid (±150 µm) sit
   near the placement-field edge at short focal lengths — cap their achievable weight.
4. One global spot size set by optics (spots overlap at 50 µm pitch — already modeled);
   verify our 30 µm spot / 50 µm pitch against the actual AMET machine optics
   (Weber tables suggest ≥68 µm 1/e² at f≥0.75 m; ~43 µm FWHM).
5. Quasi-static justification: a 49-point shape at ≥100 kHz refresh (~µs) is far
   faster than melt-pool response (~2 ms, below) — the effective-static-intensity
   assumption in `thermal_model.py` is experimentally defensible.

---

## 3. Frequency design rules — melt-pool response time constants

From the IFSW keyhole-dynamics welding papers (all Civan OPA6, keyhole-mode welding):

| Observation | Value | Source |
|---|---|---|
| Keyhole tracks beam shape 1:1 | at 50 Hz | Wagner Procedia CIRP 111 (2022) 448 |
| Transverse width follows shape | only ≲10 Hz | Sawannia Procedia CIRP 124 (2024) 430 |
| Depth follows periodically | ≤100 Hz (ceiling ~250 Hz) | Sawannia 2024 |
| Capillary re-forms after shape switch | **~2 ms** | Sawannia 2024 |
| Capillary quasi-stationary (sees average) | at 10 kHz | Sawannia 2024 |
| Keyhole stays permanently open | >1 kHz pattern freq | Volpp J. Laser Appl. 36 (2024) |
| Cu process stabilization threshold | **≥60 kHz** | Fujio (SSRN preprint, Osaka/IFSW) |
| Mid-band frequencies can be WORSE | 10 kHz → 3.7× more spatter than static | Fujio |

The melt pool is a **low-pass filter** (τ ≈ 2 ms in 5 kW welding). Above ~10 kHz the
process responds only to time-averaged intensity. Caveat for LPBF: our melt pool is
much smaller/faster (τ ~ L²/α with L ~ 100 µm → ~ms in 316L), so the required shaping
frequency is higher — worth one sentence + estimate in the paper. The **non-monotonic
frequency response** (Fujio: 10 kHz worst, 60+ kHz best; ~70% spatter reduction vs
static at ≥60 kHz) is a design trap any temporal extension of the ML pipeline must
respect.

Welding-side benchmarks: Al lap welds at up to 18 m/min with >10 kHz shaping,
penetration within 20% over 6–15 m/min (Prieto Procedia CIRP 94, 2020); hollow shapes
(spiral/infinity/ring) consistently best (Prieto; Dittrich Procedia CIRP 124, 2024 —
small simple shapes beat large complex ones on Cu); melt-flow shaping can raise
thermal efficiency up to +32% via keyhole-shape drag effects (Wagner Procedia CIRP
124 (2024) 574, OpenFOAM reduced model — candidate mid-fidelity forward model).

---

## 4. LPBF evidence: ring beams, spatter, dimensionless frameworks

### 4.1 The TUM trilogy (Grünewald/Wudy, nLight AFX-1000, 316L)

- **Metals 2021** (foundation): process window Gaussian 450 W / 1100 mm/s → ring
  **1050 W / 1700 mm/s**; ~2× productivity (V_m 12→20 mm³/s); optimum area energy
  2–4 J/mm² profile-independent; equal line energy → equal molten area (energy
  coupling shape-invariant).
- **AIME 10, 100164 (2025)** (scaling laws, single tracks + open dataset): the key
  reusable result — melting-mode thresholds collapse across ALL profiles using
  **H_max/h_s = (Φ/2)·(H/h_s)** where **Φ = peak-to-mean intensity ratio** =
  I_peak·πw₀²/P and H/h_s is Hann/King dimensionless enthalpy (with latent heat,
  2nd-moment spot size mandatory for non-Gaussian beams):
  **conduction→transition at H_max/h_s = 6.25 ± 0.85; transition→keyhole at
  8.65 ± 0.30.** Φ generalizes to ANY beam shape — we can compute it for every CBC
  design and use it as (i) a regime classifier, (ii) an ML input feature,
  (iii) a design constraint. Their supplementary single-track database is reusable.
- **AML 14, 100284 (2025)** (dimensionless 3-D process maps, cubes): map axes
  (H/h_s)/h* vs Pe·h* (h* = hatch/w₀, Pe = 2w₀v/α). Lack-of-fusion and
  balling/protrusion boundaries are **beam-profile-independent** and conduction-
  predictable (LoF: mid-hatch melt depth ≥ 5 µm; balling: l*/w* ≤ 2.1 Kruth
  criterion); the **keyhole boundary is profile-dependent** (Gaussian keyholes below
  Pe·h* ≈ 9; heavy-ring profiles showed **zero keyhole porosity**) and NOT
  conduction-predictable. Counterintuitive: small hatch (h* < 1) + high speed widens
  the window — underexplored design axis.

### 4.2 Spatter benchmarks (to contextualize our ~13%)

| Study | Material | Comparison | Metric | Reduction |
|---|---|---|---|---|
| Grünewald JLA 35 (2023) | 316L | ring vs high-peak AFX | spatter **per fused area** | **3–4× (~65–75%)** at equal ΔH/h_s |
| Tumkur Sci. Adv. 7 (2021) | 316L | Bessel vs Gaussian | cold ejecta count | ~2× (~50%) |
| Rothfelder Procedia CIRP 111 (2022) | Ti64 + W tracer | 80% ring vs Gaussian | landed tracer area | ~75% (n=3, gas-flow confounded) |
| Fujio (preprint) | Cu welding | 60+ kHz oscillation vs static | spatter count | ~70% |
| **Ours** | 316L | "Smile" vs Gaussian | spatter density | **~13%** |

Honest framing: our 13% is modest vs headline ring/Bessel numbers, but those shapes
pay a **resolution penalty** (spot grows ~2.5–3×) and a **higher critical energy
density** (Bessel needs H/h_s ~5 vs Gaussian ~3 to densify). The Smile result shows
reduction **without** giving up a compact spot — and the ML pipeline searches for
Pareto-optimal shapes on exactly that trade-off. Also: state our normalization
(per-area vs per-track changes conclusions — Grünewald shows factor 3–4 only
crystallizes per fused area). Spatter velocity depends on enthalpy, not shape
(Grünewald) — count reduction at fixed enthalpy is the defensible claim.

Methodology to mirror (Grünewald JLA 2023): coaxial HSI 20 kfps, process-zone
masking (Otsu + dilation), LoG detection + Kalman tracking (TrackMate/Fiji),
normalize per track length AND per fused area, plot vs ΔH/h_s. Tumkur adds
plume-angle variance (Bessel steady 100–120° vs Gaussian 70–140°) and WEKA
ML-segmentation of hot vs cold spatter.

### 4.3 Mechanisms (citable chain)

Lower peak/center intensity → lower peak T, weaker vapor jet + weaker Marangoni →
(i) less powder entrainment and fewer ejecta, (ii) delayed keyhole onset
(keyholing propensity slope 6.7× Gaussian vs 1.6× Bessel at the conduction→keyhole
transition, Tumkur), (iii) suppressed balling via lower melt-pool length/width
(Plateau–Rayleigh), (iv) ring pre/post-heating reduces hot cracking + humping in Al
(Nahr JMMP 7:93, 2023). Bayat: ring's annular recoil + inward Marangoni creates a
central liquid bulge → jet → spatter at high power (ring's own failure mode — beam
shapes have shape-specific instabilities, another reason for search vs presets).

### 4.4 Microstructure (G/R link — justifies our gradient objective)

- **Galbusera Opt. Laser Technol. 174, 110649 (2024)** (PoliMi, AFX, Fe-2.9Si):
  encodes shape as (Ipk, Ir); **peak intensity → depth, ring intensity → width**;
  ANOVA: grain aspect ratio driven by ring×speed interaction (F≈102), NOT peak alone
  (p=0.31). Ring + E>1000 J/mm³ → AR 9–15 → ~5, grain-boundary density down 2–6×,
  ~2× texture index. No quantitative G/R values — qualitative only.
- **Tumkur 2021** has actual G numbers from ALE3D: Gaussian G ~1×10⁷ K/m vs Bessel
  down to 2×10⁶ K/m; solidification time +60%; ~30% smaller grains; G-R pairs pushed
  toward equiaxed (Hunt).
- For quantitative G/R→CET math cite **Shi et al. Acta Mater. 184 (2020) 284**
  (beam-shape thermal modeling), Roehling Mater. Des. 195 (2020), Hunt/Kurz–Fisher.
- Our conduction-model values (Gaussian 300 W / 400 mm/s: G ≈ 8.5×10³ K/mm = 8.5×10⁶
  K/m, cooling ≈ 5.4×10⁵ K/s) sit inside the literature band (10³–10⁴ K/mm,
  10⁵–10⁷ K/s) and are the same order as Tumkur's ALE3D Gaussian G. Real validation
  target: Shi et al. / our own IR data.

### 4.5 CBC optics constraints (Zhi et al. Sci. Rep. 7:2199, 2017 — tiled-aperture physics)

Even record tiled-aperture CBC puts only **47% (theory ≤62–70%)** of power in the
intended central lobe; the rest is structured side-lobe background that still heats
the powder bed. Fill-factor trade-off is fundamental; each spot is a truncated
Gaussian with wings; renderable patterns are band-limited by the finite beamlet
array. (Civan's dense filled-aperture approach does better — 60% main lobe — but the
same physics ceilings apply.) Justifies the realizability constraints in §2.

---

## 5. Actionable changes to `ML Beam Design/`

1. **Render realism:** add ~6% four-fold side-lobe satellites + field-envelope droop
   to `beam_shapes.py`; document power-fraction ≙ dwell-fraction (integer-quantized).
2. **Φ feature/constraint:** compute peak-to-mean ratio Φ for every design; use
   H_max/h_s = (Φ/2)(H/h_s) with thresholds 6.25 / 8.65 as a physics-based regime
   classifier alongside (or replacing) the raw peak-T proxy.
3. **Frequency constraint (future temporal work):** shape refresh must stay in the
   quasi-static band; penalize the resonant mid-band (Fujio's 10 kHz trap); shape
   complexity couples to max refresh (f_shape = f_point/N_SP, N_SP ≤ 1024).
4. **Verify grid geometry** (spot size, pitch, field) against the actual AMET DBL
   optics before exporting designs.
5. **Objectives menu from literature:** minimize G (microstructure), minimize Φ
   subject to melt-width (spatter/keyhole), plume-angle variance (if imaged),
   keep l*/w* ≤ 2.1 (balling), mid-hatch depth ≥ threshold (LoF), d/w < 0.5
   (conduction mode).
6. **Position our parameters dimensionlessly:** report H/h_s, H_max/h_s, Pe for all
   five experimental shapes + designed shapes (2nd-moment spot sizes).
7. Candidate higher-fidelity forward models, in order: Eagar–Tsai analytic (already
   ~equivalent to ours), Wagner's OpenFOAM reduced melt-flow model, ray-tracing
   absorption (Volpp), full FLOW-3D/ALE3D multiphysics (Bayat/Tumkur).

---

## 6. Per-paper index (one line each)

**Civan folder**
1. Weber et al., JMMP 9:85, 2025 — DBL physics/limits; THE hardware citation (§2).
2. Weber et al., SSRN preprint — same content; cite the JMMP version.
3. Shekel, PhotonicsViews 3/2024 — Shape Generator workflow; FWHM/steering tables.
4. Wagner et al., PhotonicsViews 3/2022 — keyhole absorptance ~shape-invariant (~66–72%); ray tracing in X-ray keyholes.
5. Dittrich et al., Procedia CIRP 124:724, 2024 — 14 kW CBC characterization; 18 shapes on Cu; small simple shapes win; dual-keyhole observation.
6. Prieto et al., Procedia CIRP 94:596, 2020 — Al e-mobility; spiral/infinity best; >10 kHz widens window; 18 m/min laps.
7. Fujio et al., SSRN preprint — Cu stabilization ≥60 kHz; ~70% spatter reduction; 10 kHz worst (non-monotonic).
8. Wagner et al., Procedia CIRP 111:448, 2022 — keyhole follows beam at 50 Hz (FFT of streak images).
9. Wagner et al., Procedia CIRP 124:574, 2024 — OpenFOAM melt-flow shaping; keyhole shape → η_th up to +32%.
10. Sawannia et al., Procedia CIRP 124:430, 2024 — ⭐ melt-pool low-pass filter; ~2 ms time constant; quasi-stationary at 10 kHz.
11. Volpp et al., JLA 36:032024, 2024 — keyhole quasi-static >1 kHz; surface tension rises above boiling (ray-tracing + X-ray).
12. Bayat et al., Addit. Manuf. 93:104420, 2024 — ⭐ closest prior work; differentiate per §1.
13. Zhi et al., Sci. Rep. 7:2199, 2017 — tiled-aperture CBC ceilings (47% central lobe).

**nLight folder**
14. Grünewald et al., Metals 11:1989, 2021 — ring doubles process window/productivity.
15. Grünewald et al., AIME 10:100164, 2025 — ⭐ Φ scaling law, H_max/h_s thresholds 6.25/8.65; open dataset.
16. Grünewald & Wudy, AML 14:100284, 2025 — dimensionless process maps; keyhole boundary is the shape-dependent one.
17. Grünewald et al., JLA 35:042009, 2023 — ⭐ spatter methodology to mirror; 3–4× per-area reduction.
18. Rothfelder et al., Procedia CIRP 111:14, 2022 — W-tracer spatter, ~75% (confounded).
19. Nahr et al., JMMP 7:93, 2023 — AW-5083 tracks; ring suppresses hot cracking/humping.
20. Tumkur et al., Sci. Adv. 7:eabg9358, 2021 — ⭐ Bessel "optothermal control"; G numbers; plume-angle metric; keyhole-resistance slopes.
21. Galbusera et al., Opt. Laser Technol. 174:110649, 2024 — (Ipk, Ir) shape encoding; ring×speed drives grain AR; no quantitative G/R.

**Fundamental folder**
22. Grünewald et al., Procedia CIRP 111:65, 2022 — AOD beam shaping; "optimal shape is objective-dependent" framing quote.
