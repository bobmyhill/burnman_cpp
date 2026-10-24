These seven tables are copied unchanged from Python BurnMan's
`burnman/data/input_geotherm` directory (source revision
`e2cfac31052273b3fa9d5ce905978e97723005e9`). Model names and column use follow
`burnman/classes/geotherm.py`:

- `brown_81.txt`: Brown and Shankland (1981).
- `anderson_82.txt`: Anderson (1982).
- `Stacey_1977_continents.txt` and `Stacey_1977_oceans.txt`: Stacey (1977).
- `Katsura_2022.txt`: Katsura (2022); temperature uncertainties remain in the
  source table, while the geotherm uses depth and temperature only.
- `Plesa_2022_Mars_V_6cm3.txt`: Plesa (2022), Mars with mantle viscosity
  activation volume 6 cm³/mol.
- `Anzellini_2013_PT.txt`: Anzellini et al. (2013), Figure S4, with the mantle
  part from Steinberger and Holme (2008); coordinates are pressure in Pa.

All depth coordinates use m and temperatures use K. Repeated coordinates in
the original tabulations are preserved.

BurnMan is distributed under GPL v2 or later; these copies are distributed
as part of burnman_cpp under GPL-3.0-or-later (see the root LICENSE).

Run `python tools/export_geotherm_data.py` after changing a table to regenerate
`src/data/geotherm_data.hpp`. The native library embeds those same tables and
requires no data paths at runtime. The generated header is committed so that
standalone C++ builds do not require Python.
