`Bertka_Fei_1997_mars_mantle.dat` is copied unchanged from Python BurnMan's
`burnman/data/input_fitting` directory at revision
`e2cfac31052273b3fa9d5ce905978e97723005e9`.

The table contains oxide weight percentages and experimental conditions from
Bertka and Fei (1997), used by `example_fit_composition.py` to fit phase mass
fractions for a Martian mantle composition. Pressures are in GPa. The original
column names, phase labels, repeated analyses and uncertainty columns are retained.

The following tables are copied unchanged from Python BurnMan's
`burnman/data/input_fitting` directory at revision
`bcfe0fa8ea0180cd2178cffe76ea985bc9098dc4`, for `example_fit_eos.py`:

- `PVT_MgO_Dewaele_et_al_2000.dat`: periclase P–V–T measurements, with
  temperature in K, pressure in GPa and unit-cell volume in angstrom³.
- `Hazen_1976_TV_periclase.dat`: periclase lattice parameters in angstroms
  at temperatures in degrees Celsius.
- `Victor_Douglas_1963_deltaH_MgO.dat`: relative enthalpies in cal/mol at
  temperatures in K. The example follows the upstream conversion to J/mol and
  temperature/enthalpy covariance choices. The header labels its reference as
  273 K; the upstream example describes enthalpy relative to 298.15 K.

`Dubrovinsky_Saxena_1997_TV_periclase.dat` is copied unchanged from the same
directory at the pinned Python BurnMan revision
`b7c54ccf7bfd8583158c6dffe210c004b9570f4d`. It contains periclase volumes and
volume uncertainties in cm³/mol at temperatures in K, used for the final
thermal-expansion comparison in `example_fit_eos.py`.

BurnMan is distributed under GPL v2 or later; these copies are distributed as part
of burnman_cpp under GPL-3.0-or-later (see the root LICENSE).
