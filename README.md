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

## Installation

### Published Python wheels

Using Python 3.10 or newer, install the published package into your chosen
Python environment:

```sh
python -m pip install --upgrade pip
python -m pip install burnman_cpp
```

A virtual environment is optional. To use one, run these commands before
installing:

```sh
python -m venv .venv
source .venv/bin/activate
```

Pip installs NumPy and SciPy automatically. Published wheels bundle the native
libraries, so a compiler and native development packages are only needed when
building from source. If no matching wheel is available, pip builds from source
and needs the dependencies below.

### Building from source

#### Native dependencies

Both the C++ library and Python source builds require:

- **C and C++17 compiler toolchain**, such as GCC or Clang: compiles the native
  library and, for Python builds, the extension module. CMake also checks for
  a C compiler when locating HDF5.
- **CMake 3.18 or newer**: configures the build, locates native dependencies and
  embeds the covariance datasets. The supplied Makefile also uses CMake for
  the data embedding step.
- **Eigen 3.4 or newer**: supplies matrix and vector operations, linear solvers
  and decompositions for compositions, reactions, solution models and fitting.
- **GSL (GNU Scientific Library)**: provides root finding, numerical
  integration, special functions and splines for equations of state and radial
  planet profiles.
- **NLopt, including its C++ header `nlopt.hpp`**: performs constrained
  optimisation for relaxed solutions, fitting, thermobarometry and phase
  stability calculations.
- **cddlib, including its GMP-backed library `cddgmp`**: converts composition
  constraints into polytope vertices for solution and assemblage calculations.
- **GMP (GNU Multiple Precision Arithmetic Library)**: provides the exact
  rational arithmetic used by cddlib and the polytope code to handle chemical
  composition constraints.
- **HDF5 1.10 or newer, with its C development headers and gzip/deflate support**:
  reads the embedded endmember energy covariance datasets for HP11, HGP18 and
  HPx_ds636. These data are embedded during the build and read in memory.

Additional tools depend on the build and checks you choose:

- **Make**: runs the supplied Makefile. CMake builds also need a build tool,
  such as Make or Ninja.
- **pkg-config**: lets the Makefile discover HDF5 compiler and linker flags.
  It is optional for CMake builds.
- **Catch2 3.4 or newer**: supplies the C++ unit test runner and performance
  benchmark framework. It is only required when building those targets.
- **Git**: downloads the source repository and the optional pinned Python
  BurnMan reference checkout.

#### Python dependencies

For the Python package, also install:

- **Python 3.10 or newer, including its development headers**: runs the package
  and lets the extension compile against Python.
- **pip**: builds and installs the Python package and its Python dependencies.

Pip installs these build and runtime dependencies automatically, as specified in
[pyproject.toml](pyproject.toml):

- **scikit-build-core 0.11 or newer** (build): connects pip's package build
  process to CMake and packages the compiled extension.
- **pybind11 3.0 or newer** (build): exposes the C++ classes and functions to
  Python, including conversions between Eigen matrices and NumPy arrays.
- **NumPy 1.23 or newer** (runtime): provides arrays for compositions, material
  properties, covariance matrices and thermodynamic grids.
- **SciPy 1.9 or newer** (runtime): provides interpolation, sparse matrices,
  optimisation and smoothing for potential grids and ASPECT table exports.

Optional Python dependencies are installed through extras:

- **pytest 7 or newer** (`test`): runs the Python test suite.
- **Matplotlib 3.6 or newer** (`examples`): plots examples, phase diagrams and
  diagnostic figures; it is also needed by the plotting tests.
- **Python BurnMan 2.1 or newer** (`reference`): enables adapters and comparisons
  with the original implementation. The reference tests below use a specific
  Git commit for reproducibility.
- **h5py 3.8 or newer** (`reference`): exports and checks the HDF5 covariance
  source files. Ordinary builds and covariance access use native HDF5 directly.

#### Install system dependencies

On Ubuntu 24.04, install the native dependencies and Python development tools:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
  python3-dev python3-pip python3-venv \
  libeigen3-dev libgsl-dev libnlopt-cxx-dev libcdd-dev libgmp-dev libhdf5-dev
```

For C++ tests and benchmarks, also install Catch2:

```sh
sudo apt-get install -y catch2
```

On macOS, install the Xcode command-line tools if needed, then install the
dependencies with [Homebrew](https://brew.sh):

```sh
xcode-select --install
brew install python cmake pkg-config eigen gsl nlopt cddlib gmp hdf5
```

For C++ tests and benchmarks, also install Catch2:

```sh
brew install catch2
```

#### Build and install the Python package

Clone the repository, then use your chosen Python environment:

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
compiler and dependency paths. If HDF5 is outside pkg-config's search path, set
`PKG_CONFIG_PATH` to the directory containing `hdf5.pc`, or pass `HDF5_CFLAGS`
and `HDF5_LIBS` explicitly to Make.

### Optional Python BurnMan reference comparisons

The reference version is recorded in [tests/reference/burnman.json](tests/reference/burnman.json).
The tests require that exact commit in a clean Git checkout. To enable the
comparisons used by GitHub CI:

```sh
git clone https://github.com/geodynamics/burnman.git build/python-burnman-reference
git -C build/python-burnman-reference checkout e2cfac31052273b3fa9d5ce905978e97723005e9
python -m pip install -e 'build/python-burnman-reference[dev]' 'h5py>=3.8'
export PYTHONPATH="$PWD/build/python-burnman-reference"
export PYTHONDONTWRITEBYTECODE=1
MPLBACKEND=Agg python -m pytest -q -ra python/tests
```

## Formatting

Install the same formatter versions used by GitHub CI:

- **clang-format**: formats C++ sources and headers.
- **Black**: formats Python sources.

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

The Python reporting tools use NumPy and Matplotlib, plus:

- **pandas 1.5 or newer**: processes timing tables and CSV reports.

Install the reporting dependencies with
`python -m pip install -r benchmarks/requirements.txt`.

## License

burnman_cpp is licensed under [GPL-3.0-or-later](LICENSE).
