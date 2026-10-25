# Pitzer–Sterner H₂O and CO₂: Figure 1 (1995)

```sh
python benchmarks/pitzer_sterner_1995/benchmark.py
```

Compares the general native `pitzer-sterner` EOS with **Figure 1 at 1600 K**
in Pitzer & Sterner, *International Journal of Thermophysics* **16**, 511–518
(1995). Both fluids use the printed **Table I coefficients**, without fitting.
The graph plots `Z = P/(rho R T)` against `rho/rho_c`.

The script saves `results/comparison.pdf` and `.png`: native curves over the
published scan and differences at 17 digitized model-curve positions. CSV files
contain the sampled isotherms and comparisons; `summary.json` records RMS and
maximum differences.
Volume inversion is checked at every digitized position.

| Comparison | H₂O | CO₂ |
| --- | ---: | ---: |
| RMS difference in Z from printed fitted curve | 0.063 | 0.088 |
| RMS relative difference from printed fitted curve | 0.66% | 0.59% |

The native curves follow the printed curves within the scale of the scan.
Error bars show reading bounds from ±2 pixels in both coordinates,
propagated through the native local slope. These comparisons check reproduction
of the published model curves at the resolution of the scan.

`reference/figure1.json` preserves Table I, the axis registration, digitized
pixel coordinates, source PDF hash and extraction settings. Registration and
digitization used the scan alone. Critical densities are not tabulated in
this paper: the plot uses [322 kg/m³ for H₂O](https://iapws.org/relguide/IAPWS95-2018.pdf)
and [10.6249 mol/L for CO₂](https://nvlpubs.nist.gov/nistpubs/ir/2026/NIST.IR.8608.pdf),
without adjusting either to the curves. Molar masses and resulting molar
critical densities are recorded in the JSON.

NumPy, Matplotlib and the built `burnman_cpp` package are required. Python
BurnMan and the original PDF are not needed to run the benchmark. To recreate
the source crop with Poppler from the supplied PDF:

```sh
pdftoppm -f 4 -l 4 -r 400 -x 610 -y 370 -W 1170 -H 1110 \
  -png -singlefile ~/Pitzer_Sterner_1995_H2O_CO2_eos.pdf \
  benchmarks/pitzer_sterner_1995/reference/figure1
```

Figure 1 tests the pressure–density model. It does not constrain absolute
entropy, enthalpy or the ideal-gas heat capacity. The benchmark therefore
supplies a declared arbitrary ideal-gas caloric reference, which does not
affect P, Z, K_T or V(P,T). The general EOS uses BurnMan's shared gas constant;
R cancels in Z when the same value is used in its numerator and denominator.
