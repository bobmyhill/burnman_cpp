# Stixrude & Lithgow-Bertelloni (2024), Figure 14

Recalculate both panels near the core–mantle boundary: shear-wave velocities
of pyrolite, harzburgite and basalt, and a map of relaxed pyrolite thermal
expansivity with phase boundaries and the prescribed geotherm.

Paper: *Thermodynamics of mantle minerals — III: the role of iron*,
[doi:10.1093/gji/ggae126](https://doi.org/10.1093/gji/ggae126), Figure 14,
printed page 1713 (PDF page 15).

[Calculated figure](results/figure14.png) ·
[Overlay on the original](results/figure14_benchmark_overlay.png) ·
[Geotherm comparison](results/geotherm_comparison.png) ·
[Velocity differences](results/velocity_residuals.png)

![Calculated Figure 14](results/figure14.png)

## Run

Install the working checkout and Matplotlib, then run from the repository root:

```sh
python -m pip install '.[examples]'
python benchmarks/stixrude_2024_figure14/benchmark.py --verbose
```

The reference-overlay PNG also requires `rsvg-convert` from librsvg. The
original paper PDF is not needed: its figure and the axis calibration are
included. Fresh outputs go to `build/benchmarks/stixrude_2024_figure14/`.
Use `--output-dir PATH` to change this. The included `results/` snapshot stays
unchanged unless explicitly selected as the output directory.

The defaults are 601 points per velocity profile and a 121-pressure by
161-temperature expansivity grid. Change these with `--points N` and
`--resolution NP NT`. The phase diagram uses 9 pressure and 17 temperature
discovery seeds, independently of the property grid resolution. Gridding
requires a fully resolved diagram, with no outstanding diagram diagnostics.
Boundary points are refined to the requested resolution before gridding.

Replot the saved snapshot without any equilibrium solves:

```sh
python benchmarks/stixrude_2024_figure14/benchmark.py --replot \
  --output-dir benchmarks/stixrude_2024_figure14/results
```

Regenerate just the original-figure overlay and velocity comparison:

```sh
python benchmarks/stixrude_2024_figure14/overlay.py
```

## Geotherm

First equilibrate 100 g of Table 1 pyrolite at 0 Pa and 1400 K, giving the
surface entropy. Hold that total entropy fixed while calculating the basal
600 km of mantle. Pressure is converted to height using the no-ocean PREM
table shared with the Figure 13 benchmark; the mantle-side CMB is at 2891 km
depth and 135.751 GPa. PREM provides the pressure–depth relation, not the
temperature profile.

With `h` in km above the CMB, prescribe

```python
amplitude = 4000.0 - isentrope_temperature_at_cmb
temperature = isentrope_temperature + amplitude * erfc(h / 180.0)
```

The amplitude is calculated from the basal equilibrium on every run. It is
never a rounded input constant. Thus `T(0) = 4000 K` even if the calculated
isentrope changes. All three lithologies follow this same physical P–T path;
their equilibrium assemblages and spin populations are calculated independently.

The paper specifies a 1400 K potential temperature and a basal thermal layer
but does not give an explicit geotherm formula. This erfc model was selected
through comparison with its green curve, as requested. `geotherm.py` separately
extracts that curve from the published embedded raster, using independently
registered axis ticks. It compares the prescribed model and fitted exponential
and erfc alternatives. The extracted curve is a reference, not an input to the
equilibrium calculations.

The prescribed 180 km/4000 K model differs from the digitized curve by about
9.2 K RMS and 23.4 K maximum. The source raster's horizontal pixel represents
approximately 16.7 K. A freely fitted erfc gives a height scale of 180.09 km
and 8.9 K RMS; a fitted simple exponential gives about 38.9 K RMS. These fits
support the chosen shape without identifying the authors' exact input formula.

Run the geotherm calculation/comparison on its own:

```sh
python benchmarks/stixrude_2024_figure14/geotherm.py
```

## Composition and properties

`../stixrude_2024.py` supplies the Table 1 compositions and the complete
ordinary SLB2024 solution/pure-phase catalogue for both figure benchmarks.
Preserve the published cation ratios, set the stated initial ferric fractions
(pyrolite 0.02, harzburgite 0.01, basalt 0.12), and normalize each bulk to 100 g.
Oxygen is closed thereafter. All three native-iron polymorphs can form, and
the explicit high/low-spin endmembers equilibrate with the other species.

- `Vs = sqrt(G_VRH / density)`, with Voigt–Reuss–Hill elastic averaging.
  No extra chemical shear-relaxation law is imposed.
- `alpha = (1/V) (dV/dT)_P,bulk` includes the equilibrium changes in phase
  proportions and compositions. It uses the existing projected Gibbs-Hessian
  response calculation, retaining the solved active composition bases.
  It is neither a numerical grid derivative nor fixed-composition expansivity.
- The map covers 800–4800 K and 0–600 km above the CMB. Black lines are
  equilibrium phase boundaries from native continuation. Field labels use
  spaces between mineral names; the separate field key includes narrow fields
  whose labels cannot fit. `fp #2` denotes a second coexisting fp composition.
- Plot alpha on the published 0–5 × 10⁻⁵ K⁻¹ colour interval, with open ends
  to show clipping. Retain the full values in the data files. No response
  values are infilled or smoothed, and failed points would remain masked.

## Comparison and validation

The calculated profiles and the main bg–ppv expansivity band closely resemble
the reference, but this is not an exact numerical HeFESTo match. In the saved
601-point profiles, RMS velocity differences from the original vector curves
are approximately 0.0040 km/s for pyrolite, 0.0051 km/s for harzburgite and
0.0362 km/s for basalt. Basalt's sharp velocity jumps differ visibly. Some
iron-related phase boundaries also differ, particularly the hcp/fcc boundary
at high temperature. These discrepancies remain visible in the benchmark;
their causes have not been established here.

The overlay retains the original expansivity colours, solid velocity curves
and black phase boundaries. Calculated velocities are dashed in matching
colours, calculated phase boundaries are cyan, and the prescribed geotherm
is dashed green. The fully recalculated alpha map is in `figure14.png/svg`.
Overlay registration uses each panel's ticks, with no shifts or fitting to
improve agreement. Velocity residuals compare the independently calibrated
original vector profiles with the calculated profiles at common heights.

All 601 points in each lithology profile and all 19,481 map points solved
without equilibrium or property diagnostics. The native thermodynamic grid
audit and balance/velocity checks are recorded in the numerical report.
The complete fresh calculation and data writing took 101.85 s on the development
machine; final figure/overlay rendering is separate. `run.log` includes progress
and timings. The 172 focused regression tests passed in 5.59 s.

Small regression tests check preservation of Table 1 cation ratios, initial
ferric fractions, the dynamically calculated erfc amplitude, arbitrary sampled
paths, and relaxed expansivity against newly equilibrated volume differences:

```sh
python -m pytest python/tests/test_figure14_benchmark.py \
  python/tests/test_equilibrium_path.py -q
```

## Saved data and provenance

| File | Contents |
| --- | --- |
| `results/figure14.npz` | Profile P/T, Vs, density, shear modulus, atomic modes, masks; map axes, alpha and masks; prescribed and isentropic geotherms. |
| `results/*_equilibria.json.gz` | Full solved species populations, active bases and validation errors for each lithology. |
| `results/expansivity_grid.json.gz` | Native property grid with all retained equilibria and its thermodynamic audit. Load with `bm.load_property_grid()` and use `bm.evaluate_property_grid()` for further properties without new solves. |
| `results/phase_diagram.json` | Resolved, refined pyrolite phase boundaries, nodes, samples and diagnostics. |
| `results/geotherm.npz` | Isentrope, prescribed geotherm, digitized reference and comparison residuals. |
| `results/geotherm_report.json` | Calculated entropy/basal temperature/amplitude, prescribed conditions and comparison fits. |
| `results/report.json` | Composition, grid/profile acceptance, validation, settings and timings. |
| `results/velocity_comparison.{csv,json}` | Pointwise and summary differences from the original vector profiles. |
| `results/overlay_registration.json` | Input hashes and independent panel-axis registrations. |
| `results/field_labels.txt` | Complete assemblage key, including narrow and coexisting-solution fields. |

Read numerical arrays with `np.load(path, allow_pickle=False)`. The map arrays
have shape `(number_of_pressures, number_of_temperatures)`; map height decreases
as pressure increases. Profiles have decreasing height and increasing pressure.
Use the saved masks when replotting. Units are SI except height (km); modes are
dimensionless atomic fractions.

`reference/figure14.svg` preserves the supplied paper's artwork in PDF-page
coordinates. `reference/expansivity.jpg` is the original embedded raster, not a
new scan. `reference/axis_registration.json` records the page, source PDF hash,
tick positions and image transform. Reference artwork and scientific results
are attributed to Stixrude & Lithgow-Bertelloni (2024).
