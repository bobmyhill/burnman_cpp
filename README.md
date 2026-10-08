# burnman_cpp

A C++ translation of [BurnMan](https://github.com/geodynamics/burnman), with
Python bindings.

## Developers

C++ authors:

* Benedict Heinen
* Bob (Robert) Myhill

Python authors (as of 2025, from the [BurnMan README](https://github.com/geodynamics/burnman/blob/39b582cd23954fabd3bdbbee6526a522b1d97bc9/Readme.md)):

* Bob (Robert) Myhill (main contributor)
* Sanne Cottaar
* Timo Heister
* Ian Rose
* Cayman Unterborn
* Benedict Heinen
* Robert Farla
* Juliane Dannberg
* Rene Gassmoeller

## Requirements

- A C++17 compiler, such as GCC or Clang.
- CMake 3.18 or newer.
- Eigen 3.4 or newer, GSL, NLopt with its C++ headers, cddlib with its GMP
  library (`cddgmp`), and GMP.
- Python 3.10 or newer and its development headers for the Python package.
- Catch2 3.4 or newer for C++ tests and benchmarks.

## Installation

### Native dependencies

On Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev \
  libeigen3-dev libgsl-dev libnlopt-cxx-dev libcdd-dev libgmp-dev catch2
```

On macOS, install the Xcode command-line tools if needed, then install the
dependencies with [Homebrew](https://brew.sh):

```sh
xcode-select --install
brew install python cmake eigen gsl nlopt cddlib gmp catch2
```

### Python package

Clone the repository:

```sh
git clone https://github.com/bobmyhill/burnman_cpp.git
cd burnman_cpp
python -m pip install --upgrade pip
```

Build and install from source:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install .
python -c "import burnman_cpp"
```

Pip installs the Python dependencies and builds the C++ extension. The native
dependencies listed above must already be installed. Re-run the installation
after changing the C++ or Python sources. Adjust `CMAKE_BUILD_PARALLEL_LEVEL`
to suit the memory available on your machine.

### C++ library

Run build and test commands from the repository root. To build the static
library without Python bindings:

```sh
cmake -S . -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DBURNMAN_BUILD_PYTHON=OFF
cmake --build build/native --parallel 2
```

The library is written to `build/native/libburnman_core.a`. For dependencies in
a nonstandard location, add `-DCMAKE_PREFIX_PATH=/path/to/prefix` when configuring.

Alternatively, the Makefile builds `bin/libburnman.a`:

```sh
make -j<N>
```

Replace `<N>` with the number of parallel build jobs, for example `make -j4`.

## Tests

### Python tests

Install the test and plotting dependencies and run the suite:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install '.[test,examples]'
MPLBACKEND=Agg python -m pytest -q -ra python/tests
```

The plotting dependencies are needed by the tests; `MPLBACKEND=Agg` lets them
run without a display. Comparisons with Python BurnMan are skipped when that
optional reference package is absent.

### C++ tests

```sh
cmake -S . -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DBURNMAN_BUILD_PYTHON=OFF \
  -DBURNMAN_BUILD_TESTS=ON \
  -DBURNMAN_WARNINGS_AS_ERRORS=ON
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

With Make, build and run the tests separately:

```sh
make -j<N> test WARNINGS_AS_ERRORS=1
./bin/run_tests
```

Make also accepts an optional `BUILD_MODE` argument. Available modes:

- `BUILD_MODE=release` (default): enables compiler optimisation for faster
  execution (`-O3 -DNDEBUG`).
- `BUILD_MODE=debug`: keeps source-line and variable information for a debugger
  and uses less optimisation, making it easier to follow the program's execution.
- `BUILD_MODE=test`: adds runtime checks for memory errors, such as reading
  outside an array, and undefined C++ behaviour. These checks help
  identify bugs but make the program slower.

Debug and test builds use separate outputs, such as `bin/run_tests_debug` and
`bin/run_tests_test`. Use `CXX`, `EXTRA_INCLUDE` and `EXTRA_LIB` to override the
compiler and dependency paths.

### Optional Python BurnMan reference comparisons

The reference version is recorded in [tests/reference/burnman.json](tests/reference/burnman.json).
The tests require that exact commit in a clean Git checkout. To enable the
comparisons used by GitHub CI:

```sh
git clone https://github.com/geodynamics/burnman.git build/python-burnman-reference
git -C build/python-burnman-reference checkout 39b582cd23954fabd3bdbbee6526a522b1d97bc9
python -m pip install -e 'build/python-burnman-reference[dev]'
export PYTHONPATH="$PWD/build/python-burnman-reference"
export PYTHONDONTWRITEBYTECODE=1
MPLBACKEND=Agg python -m pytest -q -ra python/tests
```

## Formatting

Install the same formatter versions used by GitHub CI:

```sh
python -m pip install -r contrib/utilities/requirements-format.txt
```

Check formatting without changing files:

```sh
./contrib/utilities/indent.sh --check --tracked
```

Run `./contrib/utilities/indent.sh` to format C++, headers and Python files,
including untracked sources outside ignored directories. Use `--tracked` to
restrict formatting to tracked files. The script can be called from any working
directory.

## Benchmarks

Build, run and report performance benchmarks using the instructions in
[benchmarks/README.md](benchmarks/README.md).

## License

burnman_cpp is licensed under [GPL-3.0-or-later](LICENSE).
