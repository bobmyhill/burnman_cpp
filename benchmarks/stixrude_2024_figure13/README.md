# Stixrude & Lithgow-Bertelloni (2024), Figure 13

Recalculate pyrolite phase modes, native iron and the ferropericlase low-spin
fraction along a closed-system isentrope with a surface temperature of 1600 K.
Overlay both calculated panels on the published reference figure.

Paper: *Thermodynamics of mantle minerals — III: the role of iron*,
[doi:10.1093/gji/ggae126](https://doi.org/10.1093/gji/ggae126), Figure 13,
printed page 1712 (PDF page 14).

[Calculated panels](results/figure13.png) ·
[Benchmark overlay, PNG](results/figure13_benchmark_overlay.png) ·
[Benchmark overlay, SVG](results/figure13_benchmark_overlay.svg)

![Calculated curves in blue over the published panels](results/figure13_benchmark_overlay.png)

## Run the benchmark

From the repository root, install the working checkout and plotting dependencies:

```sh
python -m pip install '.[examples]'
```

The PNG overlay uses `rsvg-convert` from librsvg (`brew install librsvg` on
macOS, or package `librsvg2-bin` on Debian/Ubuntu). The original PDF is not
needed: the reference artwork and its axis calibration are included here.

```sh
python benchmarks/stixrude_2024_figure13/benchmark.py --points 801 --verbose
```

Fresh outputs go to `build/benchmarks/stixrude_2024_figure13/`. Use
`--output-dir PATH` to choose a different directory. The checked-in `results/`
snapshot stays unchanged unless explicitly selected as the output directory.

Regenerate the overlay from saved grid data, without any equilibrium solves:

```sh
python benchmarks/stixrude_2024_figure13/overlay.py \
  --grid benchmarks/stixrude_2024_figure13/results/mode_box.npz
```

## Composition and physical definitions

The starting composition follows Table 1 (relative oxide masses in grams):

| Oxide | Table 1 amount |
| --- | ---: |
| SiO₂ | 45.00 |
| MgO | 38.88 |
| FeO | 8.05 |
| Fe₂O₃ | 0.18 |
| CaO | 3.18 |
| Al₂O₃ | 4.00 |
| Na₂O | 0.13 |
| Cr₂O₃ | 0.57 |

These rounded entries sum to 99.99 g and imply Fe³⁺/ΣFe = 0.0197231.
Preserve every cation ratio, redistribute FeO/Fe₂O₃ to exactly 0.02 as specified
in the Figure 13 caption, and normalise to 100 g. Oxygen is closed thereafter;
metal can form through internal redox reactions. The actual normalised oxide
amounts and elemental inventory are recorded in `report.json`.

- Calculate the surface equilibrium at 0 Pa and 1600 K, then hold its total
  entropy fixed while sampling pressure from 0 to 135.751 GPa.
- Use the ordinary SLB2024 solution catalogue, with all three native-iron
  polymorphs and the explicit high/low-spin endmembers. Spin populations
  equilibrate with the mineral compositions; no prescribed spin curve is used.
- Convert pressure to depth with the included no-ocean PREM table. This is a
  plotting conversion; temperature is solved from the entropy constraint.
- Atomic modes weight each mineral's solved formula-unit amount by its number
  of atoms, including oxygen. Sum coexisting copies of each candidate model.
- Sum native iron over its bcc/fcc/hcp polymorphs. Following the published
  layout, its enlarged band is measured downward from 1: the dashed boundary
  is `1 - 100 * native_iron_atomic_fraction`.
- The lower curve follows equation (24):
  `f = n(wustite_LS) / [n(wustite_HS) + n(wustite_LS)]`.
  Weight species amounts across any coexisting fp copies before taking this
  ratio. Ferric iron in the magnetite endmember is outside this ferrous spin
  ratio. Where no ferrous fp exists, save NaN and leave the curve blank.

The calculation reproduces phase modes, native iron and fp spin. Ferric-ratio
curves, spin/ferric shading, oxygen fugacity and external comparison data remain
visible in the original artwork; they are not recalculated here.

## Saved results and validation

The 21 October 2026 snapshot solved all 801 points without diagnostics. The
calculation and its own plots took 11.84 s on the development machine; overlay
rendering is separate. Maximum relative entropy, elemental mass-balance and
total-atom errors were approximately 1.1×10⁻¹⁵, 4.5×10⁻¹⁶ and 4.4×10⁻¹⁶.

Native iron peaks at atomic fraction 0.0025112 (0.25112%) near 706.8 km,
25.453 GPa and 1913.1 K. The fp low-spin fraction reaches 0.70385 at 2891 km.
It is defined at 664 of the 801 points. Temperature rises from 1600 K to
2682.14 K along the isentrope.

The modes/native-iron and spin curves show close visual agreement with much of
the reference. Visible differences remain in the lower-mantle fp proportion
and the bg–ppv coexistence interval. This benchmark retains those differences;
it does not force agreement or declare an exact numerical HeFESTo match.
The authors' unrounded calculation inputs have not been independently checked.

| File | Contents |
| --- | --- |
| `results/mode_box.npz` | Solved P/T/S/V, target pressures, PREM depths, phase names, atomic modes, molar/atomic amounts, cumulative mode boundaries, native iron, fp spin and success mask. |
| `results/equilibria.json.gz` | Full solved phase amounts, original endmember proportions, active bases and equilibrium validation errors at every point. |
| `results/report.json` | Bulk composition, run settings, numerical validation and summary quantities. |
| `results/figure13.{png,svg}` | Recalculated mode and fp-spin panels. |
| `results/figure13_benchmark_overlay.{png,svg}` | Calculated blue curves over the original published panels. |
| `results/overlay_registration.json` | Input hashes and the axis transformations used for the overlay. |
| `results/run.log` | Progress and the saved run's report. |
| `results/validation.log` | The 84 passing focused path, equilibrium, workflow and benchmark tests. |

Read the numerical arrays with `np.load(path, allow_pickle=False)`. Fractions
are dimensionless; P is in Pa, T in K, total S in J/K and total V in m³ for the
100 g bulk. Original endmember coefficients can be signed for ordering models.

## Reference registration and provenance

`reference/figure13.svg` retains the original vector artwork from the supplied
paper PDF, cropped to the figure. `axis_registration.json` records the source
PDF hash, page, frame coordinates and tick positions. The two published panels
have slightly different widths and are calibrated independently in depth and
fraction. The largest tick residuals are 0.354 km (upper) and 0.261 km (lower).
Calculated curves are not fitted or shifted to match the published curves.
The source figure's original data and colours are preserved.

`reference/prem_depth_pressure.txt` contains the mantle depth/pressure columns
of BurnMan's no-ocean PREM table, based on Dziewonski & Anderson (1981),
[doi:10.1016/0031-9201(81)90046-7](https://doi.org/10.1016/0031-9201(81)90046-7).

Small regression tests check the Table 1 cation ratios, initial ferric fraction,
spin weighting across fp copies and undefined spin ratios:

```sh
python -m pytest python/tests/test_figure13_benchmark.py -q
```
