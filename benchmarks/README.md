Performance benchmarks
======================

Scientific reference comparisons are kept alongside the timing benchmarks:

- [Pitzer & Sterner (1995), Figure 1](pitzer_sterner_1995/README.md):
  native H₂O/CO₂ compression factors at 1600 K, with an overlay on the
  published figure and comparisons with the digitized model curves.
- [de Koker, Karki & Stixrude (2013), nine liquid compositions](dekoker_2013/README.md):
  direct native EOS comparisons with supplement Figures S2–S10 and Table 1.
- [Stixrude & Lithgow-Bertelloni (2024), Figure 13](stixrude_2024_figure13/README.md):
  pyrolite phase modes, native iron and ferropericlase spin along a 1600 K
  surface isentrope, with overlays on the published panels and saved equilibria.
- [Stixrude & Lithgow-Bertelloni (2024), Figure 14](stixrude_2024_figure14/README.md):
  three lower-mantle shear-velocity profiles, relaxed pyrolite expansivity and
  phase boundaries, with a 1400 K isentrope plus a basal erfc thermal layer.

The illustrative [de Koker mixing and melting reconstruction](../examples/example_dekoker_2013.py)
is an example, separate from the nine-composition EOS benchmark.

`core/` and `eos/` contain the Catch2 performance benchmarks. They build into
their own executable and reuse the fixtures in `tests/include/`. Unit tests
build separately. The Python scripts time pure Python BurnMan, process native
XML reports, and compare timings. Generated reports go into ignored `build/`
directories.

Build and run with Make:

```sh
make -j6 benchmarks
./bin/run_benchmarks '[!benchmark]'
```

Build and run with CMake:

```sh
cmake -S . -B build/benchmarks \
  -DCMAKE_BUILD_TYPE=Release \
  -DBURNMAN_BUILD_PYTHON=OFF \
  -DBURNMAN_BUILD_TESTS=OFF \
  -DBURNMAN_BUILD_BENCHMARKS=ON
cmake --build build/benchmarks --parallel 6
ctest --test-dir build/benchmarks --output-on-failure -L benchmarks
```

CTest runs every benchmark with 20 samples, 1,000 bootstrap resamples and a
10 ms warm-up. It writes `build/benchmarks/benchmark-report.xml`. Run the
executable directly to select Catch2's default measurement settings or supply
your own `--benchmark-samples`, `--benchmark-resamples` and
`--benchmark-warmup-time` values:

```sh
./bin/run_benchmarks '[!benchmark]' \
  --reporter xml::out=build/benchmarks/benchmark-report.xml --reporter console
```

Reporting tools
---------------

Install reporting dependencies and process a native report without opening
plots or waiting for input:

```sh
python -m pip install -r benchmarks/requirements.txt
python benchmarks/parse_benchmarks.py \
  build/benchmarks/benchmark-report.xml baseline --no-plots
```

The label `baseline` creates the baseline CSV filenames used by comparisons.
Supply another label for a subsequent run. Omit `--no-plots` for interactive
plots. Use `--output-dir` to choose where CSV and PDF reports are written.

With pure Python BurnMan installed, generate its timings and compare them with
the native baseline:

```sh
python benchmarks/python_benchmarks.py
python benchmarks/cpp_python_speedup.py --save-data
```

Both commands use `build/benchmarks/` by default. The Python timer accepts
`--output-dir`, and the comparison accepts `--input-dir`.

The GitHub `Benchmarks` workflow runs on pushes, pull requests and manual
dispatch. It builds and executes the native benchmarks with warnings treated
as errors, processes their XML, and runs the Python timings against BurnMan
commit `bcfe0fa8ea0180cd2178cffe76ea985bc9098dc4`. The XML, CTest log and CSV files
are available as workflow artifacts.
