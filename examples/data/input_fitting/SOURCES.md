`Bertka_Fei_1997_mars_mantle.dat` is copied unchanged from Python BurnMan's
`burnman/data/input_fitting` directory. Its source revision is recorded as
`historical_commits.initial_snapshots_and_tables` in
[the reference policy](../../../tests/reference/burnman.json).

The table contains oxide weight percentages and experimental conditions from
Bertka and Fei (1997), used by `example_fit_composition.py` to fit phase mass
fractions for a Martian mantle composition. Pressures are in GPa. The original
column names, phase labels, repeated analyses and uncertainty columns are retained.

The following tables are copied unchanged from Python BurnMan's
`burnman/data/input_fitting` directory for `example_fit_eos.py`. Their source
revision is `historical_commits.fitting_aa_and_brosh` in the reference policy:

- `PVT_MgO_Dewaele_et_al_2000.dat`: periclase P–V–T measurements, with
  temperature in K, pressure in GPa and unit-cell volume in angstrom³.
- `Hazen_1976_TV_periclase.dat`: periclase lattice parameters in angstroms
  at temperatures in degrees Celsius.
- `Victor_Douglas_1963_deltaH_MgO.dat`: relative enthalpies in cal/mol at
  temperatures in K. The example follows the upstream conversion to J/mol and
  temperature/enthalpy covariance choices. The header labels its reference as
  273 K; the upstream example describes enthalpy relative to 298.15 K.

`Dubrovinsky_Saxena_1997_TV_periclase.dat` is copied unchanged from the same
directory, using revision `historical_commits.dubrovinsky_saxena_table` in the
reference policy. It contains periclase volumes and
volume uncertainties in cm³/mol at temperatures in K, used for the final
thermal-expansion comparison in `example_fit_eos.py`.

BurnMan is distributed under GPL v2 or later; these copies are distributed as part
of burnman_cpp under GPL-3.0-or-later (see the root LICENSE).
