C++ translation of burnman (https://github.com/geodynamics/burnman)

Requires: C++17, Eigen >= 3.4, GSL
Tests/Benchmarks require: Catch2 >= 3.4

Build the native static library (`bin/libburnman.a`): `make -j6`
Build unit tests: `make -j6 test`
Run unit tests: `./bin/run_tests`
Build performance benchmarks: `make -j6 benchmarks`
Run performance benchmarks: `./bin/run_benchmarks '[!benchmark]'`

Use `BUILD_MODE=debug` for a debug build, or `BUILD_MODE=test` for address and
undefined-behaviour sanitizers. These modes use separate build directories and
output names. Override `CXX`, `EXTRA_INCLUDE` or `EXTRA_LIB` on the command line
for nonstandard installations. Dependency headers are treated as system headers;
the project retains `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` checks.
Set `WARNINGS_AS_ERRORS=1` to make warnings fail the Make build.
On macOS, native builds target the host OS by default; set
`MACOSX_DEPLOYMENT_TARGET` explicitly when building for an older system, using
dependencies built for that system. CMake accepts `CMAKE_OSX_DEPLOYMENT_TARGET`
and `BURNMAN_WARNINGS_AS_ERRORS=ON` for the corresponding settings.
If CMake reports that `-ld_classic` is deprecated, remove `-Wl,-ld_classic`
from your shell's `LDFLAGS`; the default linker works with these builds.

Matched THERMOCALC metapelite and metabasite phase sets, including calibrated
melts and PS94 water, are available through `burnman_cpp.minerals.model_sets`
and the native `burnman/minerals/model_sets.hpp` API. See the
[model-set guide](contrib/model_sets/README.md) for usage, versions and sources.

Benchmark sources and Python timing/reporting tools live in
[benchmarks/](benchmarks/README.md). The dedicated GitHub benchmark workflow runs
the native benchmarks and pinned Python BurnMan timings and saves their reports.

Install the shared formatting tools with:

```sh
python -m pip install -r contrib/utilities/requirements-format.txt
```

Run `./contrib/utilities/indent.sh` to format C++, headers and Python files,
including untracked sources outside ignored directories. Use `--check` to check
without modifying files, or `--tracked` to restrict either mode to tracked files.
GitHub runs `./contrib/utilities/indent.sh --check --tracked` with the same pinned
versions and repository formatting configuration. The script can be called from
any working directory.
