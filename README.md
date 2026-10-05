C++ translation of burnman (https://github.com/geodynamics/burnman)

Requires: C++17, Eigen >= 3.4, GSL
Tests/Benchmarks require: Catch2 >= 3.4

Install: make
Build tests/benchmarks: make test

Run benchmarks with: ./bin/run_tests [!benchmark]
Save output to xlm also with:
./bin/run_tests [!benchmark] --reporter XML::out=./benchmark-report.xml --reporter console::out=-::colour-mode=ansi
Use parse_benchmarks.py to process xml output

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
