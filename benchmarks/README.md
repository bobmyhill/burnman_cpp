Performance benchmarks
======================

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
commit `69e700647ae7efeed91dfbce147c036bce619516`. The XML, CTest log and CSV files
are available as workflow artifacts.
