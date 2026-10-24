These eight reference tables are copied unchanged from Python BurnMan's
`burnman/data/input_seismic` directory (local source revision
`e2cfac31052273b3fa9d5ce905978e97723005e9`). Their original headers retain model
attribution and units. Models follow `burnman/classes/seismic.py`:

- `prem.txt`: PREM (Dziewonski and Anderson, 1981), with QL6 attenuation
  (Durek and Ekstrom, 1996), without an ocean.
- `STW105.txt`: STW105/REF (Kustowski, Ekstrom and Dziewonski, 2008).
- `iasp91.txt`: IASP91 (Kennett and Engdahl, 1991).
- `ak135.txt`: AK135 (Kennett, Engdahl and Buland, 1995).
- `pwave_fast.txt`, `pwave_slow.txt`, `swave_fast.txt`, `swave_slow.txt`:
  mean lower-mantle profiles from Lekic et al. (2012).

BurnMan is distributed under GPL v2 or later; these copies are distributed
as part of burnman_cpp under GPL-3.0-or-later (see the root LICENSE).

Run `python tools/export_seismic_data.py` after changing a table to regenerate
`src/data/seismic_data.hpp`. The native library embeds those same tables and
requires no data paths at runtime. The generated header is committed so that
standalone C++ builds do not require Python.
