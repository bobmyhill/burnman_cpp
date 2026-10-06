# Matched THERMOCALC model sets

These factories reproduce the published HPx metapelite and metabasite
calibrations in C++, with their matching **Holland–Powell dataset 6.2** and
**PS94 H₂O fluid**. Mixing, melt speciation, Gibbs minimisation and equilibrium
continuation use the native library. Python BurnMan is needed only for optional
reference tests.

Mineral modules use the publication year in both Python and C++:

| Module | Models | Python BurnMan source |
| --- | --- | --- |
| `HP11` | Holland & Powell (2011), dataset 6.2 | `HP_2011_ds62` |
| `SLB11` | Stixrude & Lithgow-Bertelloni (2011) | `SLB_2011` |
| `JH15` | Jennings & Holland (2015) | `JH_2015` |
| `MP14` | White et al. (2014), metapelite | `mp50NCKFMASHTO` |
| `MB16` | Green et al. (2016), metabasite | `mb50NCKFMASHTO` |
| `HGP18` | Holland et al. (2018), dataset 6.33 and melt | `HGP_2018_ds633` |
| `IG18` | Legacy 2018 igneous solids and fluid | `ig50NCKFMASHTOCr` |
| `HPx_ds636` | THERMOCALC dataset 6.36 | `HPx_ds636` |
| `IG24` | Weller et al. (2024), dry igneous | `ig51W24` |
| `IG25` | Green et al. (2025), hydrous igneous | `ig51G25` |
| `SLB24` | Stixrude & Lithgow-Bertelloni (2024) | `SLB_2024` |

Use `from burnman_cpp.minerals import MP14` or
`burnman::minerals::MP14::g()`. The years identify the published models;
the matched collections below also record their later calibration releases.

```python
import burnman_cpp as bm
from burnman_cpp.minerals import model_sets

models = model_sets.metabasite(clinopyroxene="dio")
# Or: models = model_sets.metapelite()
# Or: models = model_sets.igneous()       # G25 + ds6.36 + aqueous fluid
# Or: models = model_sets.igneous("W24")  # Dry W24 + ds6.36
bulk = bm.Composition({"SiO2": 0.050, "Al2O3": 0.015, "FeO": 0.009,
                       "MgO": 0.007, "CaO": 0.012, "Na2O": 0.003,
                       "K2O": 0.001, "TiO2": 0.001, "H2O": 0.002}, "mass")
diagram = bm.pseudosection(bulk.atomic_composition, models.phases,
                          (1.0e5, 2.0e9), (573.15, 1173.15))
saved = diagram.to_dict()
saved["model_set"] = models.to_dict()  # Record calibration and source links.
```

Every factory call creates independent, mutable phase objects. The C++ API is
`burnman::minerals::model_sets::metapelite()`, `metabasite("dio")` or `igneous()`, from
[`model_sets.hpp`](../../include/burnman/minerals/model_sets.hpp); all return a
`ModelSet` with `.phases` and calibration metadata.

| Factory | HPx release | Solids and matched melt | Pressure guidance published by HPx |
| --- | --- | --- | --- |
| `metapelite()` | 23 January 2022 | White et al. (2014); eight-component granitic melt | Below 15 kbar; cautiously to 20 kbar |
| `metabasite("dio")` | 30 January 2022 | Green et al. (2016); nine-component tonalitic melt | Below 20 kbar; cautiously to 30 kbar |

The factories cover **NCKFMASHTO**; Mn-bearing extensions of the metapelite
family are not included. Published endmember formula sizes, Gibbs increments,
pressure-dependent interactions and ideal activities are retained. In
particular, melt water has activity `h²`, olivine species have five-fold Mg–Fe
mixing entropy, and the metabasite melt's `anoL = woL + silL` associate relaxes
through atomic mass balance. Individual melts are also available as
`MP14.liq()` and `MB16.L()`, imported directly from pinned Python BurnMan.

Use **one** metabasite clinopyroxene model: `"dio"` (the default) or `"aug"`.
HPx explicitly prohibits combining them. The complete Mg-bearing ilmenite
model is included. One ternary feldspar model represents both feldspars using
coexisting solution instances; adding the equivalent `k4tr` model would
duplicate it. Multiple compositions of other solution phases are likewise
handled by the pseudosection solver's `max_phase_instances` setting.

The fluid uses Pitzer–Sterner (1994) with the Holland–Powell (2011), Table 2a,
thermal reference (`H₀ = −241810 J/mol`, `S₀ = 188.80 J/(mol K)`). This keeps
water on the solids' formation-energy scale. Liquid water in the melt uses the
matching dataset's `h2oL` endmember. The implemented PS94 fluid supports
500–1700 K and positive pressures up to 5 GPa.

The [basalt example](../../examples/example_basalt_pseudosection.py) and
[shaly metasediment example](../../examples/example_metasediment_pseudosection.py)
use these sets with their existing closed, 2 wt% H₂O bulk compositions over
300–900 °C and 0–20 kbar (calculation starts at 1 bar). The
[metasediment contour example](../../examples/example_metasediment_contours.py)
uses the same metapelite factory. **Recalculate diagrams saved with the earlier
mixed model selection** before refining them or computing new contours: their
melt basis and candidate inventory have changed.

## Sources and verification

The [HPx family overview](https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/)
requires matched model sets and datasets. The
[metapelite release](https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/metapelite-set/)
and [metabasite release](https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/metabasite-set/)
specify dataset 6.2, THERMOCALC 3.50 or later, and link their full numerical
descriptions. The papers are [White et al. (2014)](https://doi.org/10.1111/jmg.12071)
and [Green et al. (2016)](https://doi.org/10.1111/jmg.12211).

All solids, melts and fluid solutions now come from the same pinned
[Python BurnMan commit](https://github.com/geodynamics/burnman/tree/39b582cd23954fabd3bdbbee6526a522b1d97bc9).
`tools/export_example_minerals.py` exports their parameters and site expressions;
there are no separate hand-written melt definitions. The catalogue audit and
reference tests compare every public factory at two non-reference P–T states,
including Gibbs energy, entropy, volume, heat capacities and solution partial
properties. Composition derivatives are also checked by finite differences.
Independent THERMOCALC benchmarks retain their 10 J/mol absolute-energy checks.

## Igneous model sets

```python
from burnman_cpp.minerals import IG24, IG25, HPx_ds636, model_sets

models = model_sets.igneous()       # G25, hydrous or anhydrous subalkaline systems
# models = model_sets.igneous("W24")  # W24, dry alkaline/subalkaline systems
melt = IG25.liq_G25w()
fluid = IG25.fl_G25()
```

Both collections use dataset **6.36**, with the appropriate solids, melt and
pure phases. G25 includes all 14 solutions, including its eleven-endmember
silicate-bearing aqueous fluid. W24 includes all 12 solutions, including
nepheline, kalsilite, leucite and melilite, and has no hydrous fluid. They use
one feldspar solution with coexisting compositions handled by the solver.
G25's PS94 water endmember limits its fluid to 500–1700 K and positive pressures
up to 5 GPa. The 20 kbar / cautious 30 kbar metadata records the general HPx
family pressure guidance, rather than extending individual EOS domains.

The source releases are W24 (28 June 2025) and G25 (21 December 2024, uploaded
7 May 2025). Dataset 6.36 is distinct from the 2011 parameter dataset; Holland
and Powell (2011) identifies its solid/liquid equations of state. See the
[authors' igneous notes](https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/hpx-eos-igneous-sets/)
and the W24 and G25 citations stored in `ModelSet`.

Legacy `IG18` and `HGP18` catalogues remain available. The igneous collection
uses G25 because the authors withdrew the H18 melt. Upstream now supplies
separate olivine and pyroxene endmember identifiers, so the exporter no longer
repairs those collisions. The older `HGP18` melt still requires its documented
`Alsi2` site-name correction to retain unit jadeite occupancy.
