The six `example_green2025_*.py` scripts reproduce Figures S3-1–S3-6 of
[Green et al. (2025), corrigendum, Supplement 3](https://doi.org/10.1093/petrology/egae079).
They share `green2025.py`, which calls the standard native `pseudosection`,
`refine_pseudosection` and `plot_pseudosection` routines.

| Figure | Example | Rock | Temperature (°C) | Pressure (kbar) |
| --- | --- | --- | --- | --- |
| S3-1 | [klb1](example_green2025_klb1.py) | Kilbourne Hole peridotite | 800–2000 | 0–50 |
| S3-2 | [re46](example_green2025_re46.py) | Icelandic basalt | 1000–1400 | 0–12 |
| S3-3 | [at1](example_green2025_at1.py) | High-Al basalt | 850–1350 | 0–20 |
| S3-4 | [mix1g](example_green2025_mix1g.py) | Synthetic pyroxenite | 600–1600 | 0–20 |
| S3-5 | [ton101u](example_green2025_ton101u.py) | Water-undersaturated tonalite 101 | 600–1300 | 1–10 |
| S3-6 | [ton101s](example_green2025_ton101s.py) | Water-saturated tonalite 101 | 600–1300 | 1–10 |

```sh
python examples/example_green2025_klb1.py
python examples/example_green2025_ton101s.py --label-fontsize 8
```

Each writes a JSON calculation, PNG/SVG figure, Markdown assemblage key and
JSON comparison report into its own `examples/green2025_*_output/`
directory. `--check-only` checks published interior assemblages without tracing
a diagram; `--quick` uses coarse discovery; `--seeds 9` increases discovery.
`--plot-json FILE` replots a saved calculation and `--refine-json FILE`
continues its phase lines from successful equilibrium states.

The compositions use the oxide **mole amounts printed in the figure footers**,
which carry more digits than Table S3-1. They are normalised together, including
water. `FeO` is total iron and `O` is excess **atomic** oxygen:
Fe₂O₃ = 2 FeO + O, so Fe³⁺/total Fe = 2 O / FeO. The system is closed to water
and oxygen; neither water saturation nor oxygen fugacity is imposed as a buffer.
The two tonalites differ in their supplied water inventories.

All examples use `model_sets.igneous("G25")`, including the G25 melt,
dataset 6.36 and, for the tonalites, the silicate-bearing G25 aqueous fluid
with PS94 H₂O. As in MAGEMin's igneous setup, the dry bulks exclude fluid,
biotite, muscovite, amphibole, epidote and cordierite, including their formal
anhydrous corners. Its pure-phase list contains stishovite and excludes
corundum; the examples make that same selection from the general collection.
These choices are recorded in
[MAGEMin's igneous catalogue](https://github.com/ComputationalThermodynamics/MAGEMin/blob/v1.4.9/src/TC_database/TC_init_database.c)
and its
[dry-system exclusions](https://github.com/ComputationalThermodynamics/MAGEMin/blob/v1.4.9/src/TC_database/gss_function.c).
The plots start at 1 bar where the PDF displays zero pressure.

The PDF distinguishes solution branches by composition: plagioclase/alkali
feldspar, augite/pigeonite and Mg–Al spinel/magnetite/chromite/ulvospinel.
Here those are labelled `fsp`, `cpx` and `sp`. A prefix such as `2fsp` counts
coexisting compositions of the same model, rather than treating them as one
phase. All assemblage labels and fallback field numbers use the chosen font
size; the full names of numbered fields are in the separate key.

| Example | Closed fields | Liquidus isobar (kbar) | PDF liquidus (°C) | Calculated (°C) | Difference (°C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| klb1 | 27 | 25 | 1890 | 1887.16 | −2.84 |
| re46 | 17 | 6 | 1279 | 1277.40 | −1.60 |
| at1 | 54 | 10 | 1283 | 1285.27 | +2.27 |
| mix1g | 49 | 10 | 1454 | 1449.23 | −4.77 |
| ton101u | 67 | 5 | 1225 | 1223.66 | −1.34 |
| ton101s | 80 | 5 | 1001 | 999.42 | −1.58 |

Visual overlays agree closely with the main published boundaries. There are additional narrow accessory fields: for example, KLB1 has a trace-ilmenite field near 800 °C and 20 kbar, and MIX1G resolves very small two-clinopyroxene fields near 980 °C and 8 kbar.
