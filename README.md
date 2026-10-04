C++ translation of burnman (https://github.com/geodynamics/burnman)

Requires: C++17, Eigen >= 3.4, GSL, cddlib (GMP build), GMP, NLopt
Tests/Benchmarks require: Catch2 >= 3.4

Install: make
Build tests/benchmarks: make test

Run benchmarks with: ./bin/run_tests [!benchmark]
Save output to xlm also with:
./bin/run_tests [!benchmark] --reporter XML::out=./benchmark-report.xml --reporter console::out=-::colour-mode=ansi
Use parse_benchmarks.py to process xml output

## Python wrapper

The `burnman_cpp` package exposes the C++ implementation through pybind11.
It includes compositions, minerals, assemblages, ideal and symmetric/asymmetric regular
solutions, and constrained equilibration. This package has its own import name
and can be installed alongside the original Python `burnman` package.

Install with Python >= 3.10, a C++17 compiler, CMake >= 3.18, Eigen >= 3.4
GSL, cddlib's `cddgmp` library, GMP and NLopt. The existing Makefile remains available
for C++ builds.

```sh
# macOS native dependencies
brew install eigen gsl cddlib gmp nlopt cmake
# Debian/Ubuntu native dependencies
# sudo apt install libeigen3-dev libgsl-dev libcdd-dev libgmp-dev libnlopt-cxx-dev cmake g++

python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/solution_equilibrium.py
```

For development and tests:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Editable installs require reinstalling after changing C++ bindings or sources.
Nonstandard Eigen/GSL installations can be supplied through
`CMAKE_PREFIX_PATH`, or `-Ccmake.define.Eigen3_DIR=/path/to/eigen/cmake` and
`-Ccmake.define.GSL_ROOT_DIR=/path/to/gsl`. Python build dependencies are
installed automatically by pip. The build follows pybind11's documented
[CMake and scikit-build-core packaging approach](https://pybind11.readthedocs.io/en/stable/compiling.html).

### Compositions

`Composition` stores formula components and converts between mass in kg,
component amounts in mol, and elemental amounts in mol of atoms. Its main API
matches Python BurnMan; calculations and nonnegative least-squares basis
changes run in C++ and require neither Python BurnMan nor SciPy.

```python
import burnman_cpp as bm

bulk = bm.Composition({"MgSiO3": 0.5, "FeSiO3": 0.5}, unit_type="molar")
bulk.renormalize("mass", "total", 1.0)
bulk.add_components({"Al2O3": 0.1}, unit_type="molar")
bulk.change_component_set(["MgO", "SiO2", "FeO", "Al2O3"])
bulk.print("molar", significant_figures=3, normalization_amount=100.0)
print(bulk.mass_composition, bulk.atomic_composition)
```

Pass `bulk.atomic_composition` to `equilibrate()` or `pseudosection()`; include
zero amounts for any additional elements required by the candidate phases.
`normalize=True` in the constructor makes the supplied component amounts sum
to one in the input unit. `"weight"` aliases `"mass"`. Properties return copies;
use `add_components()` and `renormalize()` to modify the object. Addition,
subtraction, scalar multiplication and division support signed composition
differences. `remove_null_components(tol)` uses a molar tolerance.

`change_component_set()` preserves the elemental inventory in a nonnegative
new basis, with a relative residual tolerance of `1e-10`. It accepts redundant
bases and raises `ValueError` without modifying the object if the inventory
cannot be represented. Formulae support decimal/fractional counts, e.g.
`Mg0.5Fe1/2SiO3`, using the same atomic mass table as Python BurnMan.

`print()` and `format()` leave the stored composition unchanged; as in Python
BurnMan, `significant_figures` controls decimal places. The native
`file_to_composition_list(fname, unit_type, normalize)` reads whitespace tables
whose header lists formula components followed by `Comment` and returns
compositions plus comment word lists. See [the example](examples/composition.py).

The C++ class is available in `burnman/core/composition.hpp`. Its
`ComponentAmounts` type is an ordered vector of `(formula, amount)` pairs,
and properties are available through `get_*()` methods:

```cpp
#include "burnman/core/composition.hpp"

burnman::Composition bulk({{"MgSiO3", 0.5}, {"FeSiO3", 0.5}}, "molar");
bulk.change_component_set({"MgO", "SiO2", "FeO"});
auto atoms = bulk.get_atomic_composition();
```

### Solutions

Create endmember minerals from parameter dictionaries or `MineralParams`.
The dictionaries accept the original BurnMan names `n`, `Debye_0` and EOS
strings such as `"slb3"`; the C++ names `napfu`, `debye_0` and `EOSType`
enums are also accepted. `Cp` accepts a sequence of four coefficients.
Mineral parameters are validated and copied at construction. The `params`
property returns a snapshot; construct a new mineral to change parameters.

Given `periclase` and `wuestite` minerals, as constructed in
[the complete example](examples/solution_equilibrium.py):

```python
import burnman_cpp as bm

model = bm.SymmetricRegularSolution(
    [(periclase, "[Mg]O"), (wuestite, "[Fe]O")],
    energy_interaction=[[13.0e3]],
)
oxide = bm.Solution(model, [0.8, 0.2], name="Ferropericlase")
oxide.set_state(25.0e9, 2000.0)
print(oxide.density, oxide.molar_gibbs, oxide.activities)
oxide.set_composition([0.6, 0.4])
```

`IdealSolution(endmembers)` and
`AsymmetricRegularSolution(endmembers, alphas, energy_interaction, ...)`
are also available. Interaction arguments are upper triangular lists with
row lengths `n-1, n-2, ..., 1`. Optional `volume_interaction` and
`entropy_interaction` have the same format. Each `Solution` owns a copy of
its model's endmembers, so sharing a model between solutions preserves
independent pressure and temperature states.

Material properties include volume, density, Gibbs/Helmholtz/internal energy,
entropy, enthalpy, heat capacities, elastic moduli, thermal expansivity and
seismic velocities. Solutions also expose activities, activity coefficients,
excess properties, partial molar properties and Gibbs/entropy/volume Hessians.
Properties are available as Python attributes and C++-style `get_*()` methods.
Call `set_state()` before reading state properties. `set_composition()` clears
the cached solution properties. Eigen arrays and matrices return NumPy copies.

All quantities use SI units: Pa, K, kg/mol, m³/mol and J/mol.

### Equilibration

```python
assemblage = bm.Assemblage([oxide], [1.0])
result = bm.equilibrate(
    {"Mg": 0.6, "Fe": 0.4, "O": 1.0},
    assemblage,
    [bm.PressureConstraint(25.0e9), bm.TemperatureConstraint(2000.0)],
    store_iterates=True,
)
solve = result.sol_array[0, 0]
print(solve.success, solve.message, solve.F_norm)
print(result.prm.parameter_names, solve.x)
```

Supply exactly two equality constraints plus one per free compositional
vector. Equilibration uses flat assemblages of mineral and solution phases.
Bulk composition dictionaries must include every assemblage element,
including elements with zero amount. `PressureConstraint` uses Pa and
`TemperatureConstraint` uses K. `EntropyConstraint` uses total entropy in J/K
and `VolumeConstraint` uses total volume in m³ for the specified bulk amount.
`PTEllipseConstraint`, `LinearXConstraint`, `PhaseFractionConstraint` and
`PhaseCompositionConstraint` are also exposed. Build the latter two with the
parameters from `get_equilibration_parameters(assemblage, composition, free_vectors)`.

A list of constraints in an argument generates a Cartesian grid:

```python
result = bm.equilibrate(
    {"Mg": 0.6, "Fe": 0.4, "O": 1.0}, assemblage,
    [[bm.PressureConstraint(p) for p in (10.0e9, 25.0e9)],
     [bm.TemperatureConstraint(t) for t in (1500.0, 2000.0)]],
)
assert result.sol_array.shape == (2, 2)
```

`sol_array` is a NumPy object array of `DampedNewtonResult` objects containing
`success`, `code`, `message`, `x`, `F`, `J`, `F_norm`, `n_iterations` and optional
`iteration_history`. Check `success` for each solve. `tol`, `max_iterations`
and `verbose` control the native solver.

Equilibration mutates the assemblage and its phase objects. The C++ solver
does not yet store separate assemblage snapshots. Apply a selected grid point
using `set_composition_and_state_from_parameters(assemblage, solve.x)`.
`get_parameter_vector()` and `get_endmember_amounts()` expose the corresponding
vectors. Assemblages default to one mole and Voigt-Reuss-Hill averaging; use
`n_moles`, `set_fractions()` and `set_averaging_scheme()` to adjust them.
After changing a phase directly, call `assemblage.reset_cache()` before
querying cached aggregate properties. Python EOS subclasses are not supported.

### BurnMan's equilibration example

[examples/example_equilibrate.py](examples/example_equilibrate.py) reproduces
all eight scenarios from the pure Python example: aluminosilicate boundaries,
orthopyroxene ordering, the pyrope–grossular solvus, Mg–Fe partitioning, fixed
olivine composition, upper/lower mantle equilibria, and olivine polymorphs.
It uses the original bulk compositions and sampling grids. Mineral factories,
site chemistry, solution models, property corrections, thermodynamics, and
equilibrium calculations run in C++. Python selects constraints and plots the
results. The example requires NumPy and Matplotlib; it does not import Python
BurnMan, SciPy, CVXPY, or a Python polytope solver.

```sh
python -m pip install -e '.[examples]'
python examples/example_equilibrate.py
python examples/example_equilibrate.py --case lower_mantle --show
python examples/example_equilibrate.py --quick --no-plots
```

The default saves six PNG figures to `build/example_equilibrate/`. Select an
output directory with `--output-dir`; `--show` also opens the plots. The two
single-state examples print phase proportions and compositions. Each solve
reports convergence, and unsuccessful grid points are omitted from plots,
as in the original example.
The solvus endpoint is retried with a fresh starting composition if its
continuation step fails as the two phases merge.

The implemented subset of the mineral datasets is available through native
factories, which return independent objects:

```python
from burnman_cpp.minerals import SLB_2011

olivine = SLB_2011.mg_fe_olivine()
olivine.set_state(10.e9, 1500.)
print(olivine.density)
```

The native catalogue also includes three fixed reduced models for the example's
chemical systems: `JH_2015.mg_fe_orthopyroxene()`,
`SLB_2011.pyrope_grossular()`, and `SLB_2011.mg_fe_bridgmanite_binary()`.
These remain available for compatibility. The example now constructs its
reduced models at runtime with the general C++ polytope simplifier.
The checked-in C++ data is generated by `tools/export_example_minerals.py`;
that developer utility uses the reference BurnMan package when regenerating
the catalogue and is not invoked during installation or execution.

For existing Python BurnMan objects, the optional
`burnman_cpp.adapters.from_burnman()` converter remains available. Install
`.[reference]` to use it or run reference comparisons; the native example
does not call this adapter.

### Polytope-based solution simplification

`simplify_composite_with_composition()` enumerates feasible assemblage amounts
with [cddlib](https://github.com/cddlib/cddlib), using GMP rational arithmetic,
and removes phases that cannot occur at the specified bulk composition.
Each remaining solution is restricted to the smallest site-occupancy face
containing its feasible compositions. Independent physical vertices of that
face form a new endmember basis. Signed basis coordinates are permitted;
physical site occupancies must remain nonnegative. All of this work runs in
C++, including mineral construction and interaction-parameter transformations.

```python
import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011

garnet = SLB_2011.garnet()
original = bm.Assemblage([garnet], [1.0])
reduced = bm.simplify_composite_with_composition(
    original, {"Mg": 1.5, "Ca": 1.5, "Al": 2.0, "Si": 3.0, "O": 12.0}
)
assert reduced.phases[0].n_endmembers == 2  # pyrope and grossular
print(reduced.phases[0].basis)  # new endmembers in the original basis
```

The input assemblage is preserved. Returned phases and solution models own
independent state. Ideal and symmetric/asymmetric regular solutions are
supported. A solution reduced to one endmember becomes a mineral. Missing
bulk elements mean zero amount; infeasible compositions raise `ValueError`.
Fixing an interior bulk ratio retains the full solution model if all its
physical endmembers are needed. Returned phase fractions start uniformly;
use `equilibrate()` to determine equilibrium proportions.

The algorithm selects the whole feasible site-occupancy face, rather than
solving Python BurnMan's auxiliary minimum-norm endmember mixture problem.
It can choose a different basis for the same physical space. Gibbs energy,
entropy, volume and their compositional derivatives are preserved by the
basis transformation. Solution `basis` rows express the transformed
endmembers in the original basis; for a composition `p`, the original
coordinates are `basis.T @ p`.

`composite_polytope_at_constrained_composition()` exposes the feasible vertices
as original endmember amounts. `solution_polytope_from_endmember_occupancies()`
exposes physical vertices in independent-endmember coordinates and their
mapped site occupancies. `MaterialPolytope(equalities, inequalities)` accepts
rows `[b, a...]` denoting `b + a*x == 0` or `>= 0`, and exposes `vertices`,
`rays`, `lineality`, `is_empty` and `is_bounded`. Unbounded or empty assemblage
polytopes cannot be simplified.

Input numbers are approximated by small rational fractions within an absolute
`rational_tolerance` of 1e-12. Pass zero to retain exact binary floating-point
values; redundant chemical balances may then disagree by roundoff. The
separate `tolerance` (default 1e-10) controls phase removal, face membership
and numerical rank selection. cddlib calls are serialized because its
arithmetic state is global; Python's GIL is released during native work.

`transform_solution_to_new_basis(solution, basis, molar_fractions=..., ...)`
also exposes the native transformation directly. Rows must be independent,
sum to one and describe valid physical endmembers. Derived minerals include
standard-state energy, entropy and volume corrections, including changes in
configurational entropy. A single-row basis returns a mineral.

Native minerals also accept `set_property_modifiers([(name, parameters), ...])`
using BurnMan's modifier dictionaries. `CombinedMineral(minerals, molar_amounts,
energy_adjustment=[0, 0, 0], name="...")` creates an endmember from a signed
linear combination, with optional `[delta_E, delta_S, delta_V]` corrections.
Transformed solution models permit signed endmember coordinates provided
their site occupancies remain nonnegative.

`equilibrate(..., tol=...)` accepts a scalar absolute Newton-step tolerance or
a vector in `result.prm.parameter_names` order. The example uses 1 Pa, 1e-6 K,
and 1e-9 for phase amounts and compositions, matching the Python example's
separate tolerances. A single small tolerance in Pa can be below numerical
precision at mantle pressures.

For C++ library-only builds through CMake:

```sh
cmake -S . -B build/cmake -DBURNMAN_BUILD_PYTHON=OFF
cmake --build build/cmake
```


### Isochemical phase diagrams (pseudosections)

`pseudosection()` accepts a closed elemental bulk, candidate minerals/solutions,
and pressure/temperature ranges in SI units. The Python function calls C++
with the GIL released. Native cddlib/GMP Gibbs LPs, multistart NLopt solution
minimisation and `equilibrate()` perform all numerical work.

```python
import burnman_cpp as bm
from burnman_cpp.minerals import HP_2011_ds62 as HP

settings = bm.PseudosectionSettings()
settings.pressure_seeds = settings.temperature_seeds = 7
result = bm.pseudosection(
    {"Al": 2., "Si": 1., "O": 5.},
    [HP.andalusite(), HP.ky(), HP.sill()],
    pressure_range=(1.e5, 1.e9),
    temperature_range=(500., 1200.),
    settings=settings,
)
print(result.resolved, result.diagnostics)
for edge in result.boundaries:
    print(edge.zero_phase, [(p.pressure, p.temperature) for p in edge.points])
```

The algorithm uses a seed grid to discover assemblages. It refines finite
pseudocompounds with continuous tangent-plane minimisation, then checks native
equilibrium roots against every candidate. It allows multiple coexisting
instances of a solution. Positive site occupancies, including signed ordering
coordinates, define the physical composition domain. Inputs and their models
are copied; calculations do not mutate them.

Each field edge has a zero phase amount. A scaled Jacobian null direction
predicts the next point, and `equilibrate()` corrects it with a zero-amount and
an arclength constraint. At a junction it solves two zero-amount constraints.
The adjoining searches add/drop one phase and test phase replacements; a
normal double-zero junction gives four branches. Chemical feasibility,
stability and the Jacobian's projection into P,T prune branches at reduced
variance. The aluminosilicate triple-point test gives three physical lines.
Domain endpoints are solved with a fixed pressure or temperature constraint.
The predictor carries forward the last accepted amounts and compositions,
shortening the composition step to respect site occupancies. Trivial roots
with identical solution copies are rejected, with smaller steps near solvi.
`max_recovery_passes` (default 2) retries unfinished lines from their last
accepted states and explores branches at newly discovered junctions.
If a double-zero solve fails, phase entry is bracketed on the existing
equilibrium boundary before introducing the new phase. An ill-conditioned
Newton solve retries with parameter and equation scaling; reaction affinities,
mass balance, site bounds and imposed constraints still require verification.
Fixed-P,T field solves drop phases blocked at zero amount, reuse the surviving
phase compositions and repeat equilibrium and stability checks. Boundary
continuation retains its constrained zero phase.
Verified junctions retain their double-zero Jacobians. Points admitted slightly
past an event by the affinity tolerance are trimmed at that junction so the
displayed line cannot double back into an artificial open spur.
Line geometry is not an interpolation of raster labels.

Results expose `fields`, `boundaries`, `nodes`, `samples`, `phase_names`,
`diagnostics`, `resolved`, `equilibrium_solves` and `minimization_calls`.
Boundary points include phase amounts/compositions, mass-balance error,
minimum candidate affinity and equilibrium residuals. Phase IDs index
`phase_names`; `candidate_index` maps back to the input list. Node
`gibbs_variance` is C - number_of_coexisting_phases + 2; `pt_nullity` describes
the local node in the two-dimensional diagram. Field sample indices refer to
`samples`, and line `side_a`/`side_b` identify neighbouring assemblages.
Instances of each solution receive contiguous IDs within an assemblage; a
single remaining instance uses the candidate's first ID. Fields group samples
with the same coexisting phase families and instance counts, including
disconnected regions with that assemblage.
`stable_equilibrium()` exposes the same native stability search at one P,T;
its `equilibrium_error` is the largest reaction affinity in J/mol.
Samples with `is_field_verification=True` were solved explicitly inside a
constructed face. Their classification uses floating-point precision rather
than the geometry snapping distance, allowing very thin verified fields to
be coloured. This flag is preserved in the examples' JSON output.

Settings control seed counts, scaled continuation `step` (fraction of the
P/T domain), minimum step, number of minimisation starts, maximum coexisting
solution instances, iteration/search limits and affinity/mass-balance
thresholds. The standard C++ and Python defaults include the improvements
developed for the basalt, metasediment and pyrolite examples: active composition
faces with full-polytope stability checks, diagnosed EOS-domain exclusions,
`min_step=1.e-7`, up to 150 refinement iterations, 1000 lines and two recovery
passes. These are ceilings; converged searches stop earlier. Warm starts,
scaled corrector retries, phase-entry bracketing, critical-point continuation,
phase-rule junction matching, frame closure, interior field verification and
adjacent-field merging are shared routines. No example-specific numerical
patches are needed. The examples select their bulk, candidates, ranges, seed
density and model-domain policy, and use the common serializer and plotter.
Failed seeds, continuation failures and search limits remain
explicit. `resolved` means the attempted searches passed their numerical
checks; finite seeds and multistart minimisation do not certify exhaustive
field discovery or a global minimum. Increase seed density and minimisation
starts to assess narrow/disconnected fields and nonconvex models. Supported
solution models are ideal and symmetric/asymmetric regular solutions.

`plot_pseudosection()` draws the traced lines and colours closed fields by
the number of coexisting phases, with a discrete integer colourbar:

```python
fig, ax = bm.plot_pseudosection(result, phase_aliases={"g": "gt"},
                               label_fontsize=8,
                               label_key_path="pseudosection_assemblages.md")
fig.savefig("pseudosection.svg")
```

Polygon construction, intersection splitting, holes and assemblage assignment
run in C++; Python renders the polygons with optional Matplotlib. The default
merges adjacent identified regions with exactly the same assemblage and removes
their internal line segments, including divisions caused by overlapping chords.
Each merged region receives one label. Different assemblages, different solution
multiplicities, disconnected regions and unresolved faces remain separate.
Use `merge_fields=False` in either plotting or polygon construction to inspect
the original subdivision. The default axes use °C and kbar.
Pass `pressure_unit="GPa"` or `temperature_unit="K"` to
change units, and `ax=...` to draw into an existing figure. The calculation
frame closes fields reaching its edges; `close_domain=False` fills only
loops enclosed entirely by phase lines. Each coexisting solution instance
counts as a phase. Closed regions use the verified interior sample furthest
from their outline, falling back to agreeing boundary labels. Regions with
unfinished internal boundaries or unknown phase counts stay uncoloured.
`pseudosection_field_polygons(result)` exposes the native rings, holes, phase
counts, sample indices, `has_open_boundary`, interior `label_position` and
diagnostics. Each polygon's `source_regions` lists its zero-based original
region indices; diagnostics also refer to those original indices. Native
`boundary_segments` contains the visible edges after merging. The plotted
geometry is available as `ax.pseudosection_geometry`. Identified
closed fields are labelled with their assemblages; positions are computed
in C++ to avoid holes and concave edges. `phase_aliases` controls abbreviations,
and `label_assemblages=False` hides labels. `2hb` denotes two solution instances;
every assemblage and numeric label uses the same user-selected `label_fontsize`
in points. The rendered text box, including its background, must fit inside
the field and avoid holes. Names that do not fit become unique field numbers;
numbers that cannot fit use fine leader lines without shrinking the font.
`label_key_path` writes a separate Markdown list of numbered assemblages
(default `pseudosection_labels.md`). Pass `None` to skip file output and use
the dictionary `ax.pseudosection_label_key` directly.
When boundary labels disagree about an assemblage but agree about its phase
count, the field can be coloured while its assemblage label is omitted, with
a diagnostic. Unpaired endpoints near solution coalescence are marked in red;
fields containing them remain unresolved.
Both functions accept a saved result dictionary as well as a native result.

`refine_pseudosection(bulk, candidates, previous, settings=None)` resumes saved
unfinished lines without repeating the initial seed grid. Pass the same bulk,
candidate models/names and `max_phase_instances` used in the original search.
`previous` can be a native result or its full saved dictionary. Omitting
`settings` reuses `previous.settings`, including phase-ID stride, tolerances
and required EOS models. Supply explicit settings to change search limits.
The C++ overload without settings has the same behaviour.

`result.to_dict()` and `PseudosectionResult.from_dict(data)` provide shared,
complete JSON serialization. They retain accepted compositions and amounts,
critical modes, fields, diagnostics, EOS exclusions and the calculation
settings; all three examples use this API. `result.settings` returns a copy.
`PseudosectionSettings.to_dict()` / `.from_dict()` expose the same settings
schema and reject unknown option names. Legacy example JSON files remain
readable; files without a settings record use current defaults, with the
older pyrolite EOS-policy flags honoured when present.

```python
import json
from pathlib import Path

Path("diagram.json").write_text(json.dumps(result.to_dict(), allow_nan=False))
saved = json.loads(Path("diagram.json").read_text())
continued = bm.refine_pseudosection(bulk, candidates, saved)
```

Recovery uses accepted endpoint compositions and the outgoing tangent; parent
junction states seed new branches. At solution coalescence, the native solver
replaces the two copies by one and uses zero Gibbs curvature and zero third
directional derivative as `equilibrate()` constraints. A composition-separation
constraint seeds the adjoining solvus arm, before P/T continuation resumes.
`node.critical_mode` records the verified composition direction. Saved
identical-copy false junctions are rewound to distinct compositions. Completed
overlapping curves are consolidated after checking equilibrium equivalence;
ambiguous closed fields use nearby verified states and native stability checks.

Run the basalt example with exactly 2 wt% H2O in a closed system:

```sh
python -m pip install '.[examples]'
python examples/example_pseudosection.py --output-dir pseudosection_output
# Faster initial search:
python examples/example_pseudosection.py --quick --no-plots
# Redraw saved results with coloured fields, without new equilibrium solves:
python examples/example_pseudosection.py --plot-json pseudosection_output/basalt_pseudosection.json --label-fontsize 8
# Continue unfinished lines from saved equilibrium states:
python examples/example_pseudosection.py --refine-json pseudosection_output/basalt_pseudosection.json
```

The example spans 300–900 °C and displays 0–20 kbar, starting calculations at
1 bar because a free fluid has a singular chemical potential at zero pressure.
It uses `plot_pseudosection()` to write PNG and SVG figures with native field
edges and phase-count colours, and a JSON record of
states, topology and diagnostics, plus `basalt_pseudosection_assemblages.md`
listing the numbered fields. Fields carry assemblage abbreviations,
including `gt pl q`; unfinished endpoints are marked in red.
All three pseudosection examples use native `Composition` objects and pass
their `atomic_composition` to calculation and refinement routines. Input
masses are converted from grams to kilograms for `Composition`; the 100 g
basalt and metasediment compositions include exactly 2 g H2O and separately
specified FeO/Fe2O3. All water and oxygen remain in the bulk. The pyrolite
example adds FeO and Fe2O3 in molar amounts to set its initial ferric ratio,
then renormalizes to 0.1 kg. Formula conversions use BurnMan's shared atomic
mass table.

The shaly metasediment example uses the same 300–900 °C and 0–20 kbar range
and closed-system 2 wt% H2O assumption:

```sh
python examples/example_metasediment_pseudosection.py --output-dir metasediment_output
python examples/example_metasediment_pseudosection.py --quick --no-plots
python examples/example_metasediment_pseudosection.py --plot-json metasediment_output/metasediment_pseudosection.json --label-fontsize 8
python examples/example_metasediment_pseudosection.py --refine-json metasediment_output/metasediment_pseudosection.json
```

Its illustrative bulk, in wt% of a wet rock, is SiO2 61.5, Al2O3 20, FeO 5,
Fe2O3 1, MgO 2.5, CaO 1, Na2O 2, K2O 4, TiO2 1 and H2O 2 (total 100).
Native `mp50NCKFMASHTO` factories supply metapelite solutions, including
white mica (`ms`), biotite (`bi`), chlorite (`chl`), chloritoid (`ctd`),
staurolite (`st`), cordierite (`cd`) and garnet (`gt`). Pure andalusite,
kyanite and sillimanite, oxides, HGP2018 melt and PS94 H2O are also candidates.
One ternary feldspar model (`fsp`) represents plagioclase and alkali feldspar
through coexisting compositions; `2fsp` means two feldspar phases. The
thermodynamically identical `k4tr` factory is not added as a duplicate candidate.
The example writes `metasediment_pseudosection.json`, PNG and SVG figures,
and `metasediment_pseudosection_assemblages.md`, with adjacent identical fields
merged in C++ and the chosen uniform label font. The melt/fluid combination
is illustrative, as in the basalt example. Staurolite's fractional formula
unit is exported with exact molar scaling rather than rounding its atom count.

Native `mb50NCKFMASHTO` factories provide the metabasite solid solutions;
`HGP_2018_ds633.silicate_melt()` provides an 11-endmember, Cr-free hydrous melt.
The melt uses the native `hp_tmtL` liquid EOS. Its jadeite pseudo-species is
spelled `Alsitwo`: the reference file's `Alsi2` is parsed as occupancy two,
although its original THERMOCALC definition specifies `pjd 1`. This correction
is tested against Gibbs gradients and Hessians. It intentionally changes that
erroneous reference-model mixing term.

`water_fluid()` implements the Pitzer–Sterner (1994) density/Helmholtz EOS and
chooses the minimum-G mechanically stable density root. It uses the NIST
Shomate ideal-gas thermal reference for 500–1700 K and supports positive
pressure up to 5 GPa. Numerical thermal derivatives are checked against
thermodynamic identities. The example combines HP dataset 6.2 metabasite
solids, HGP 2018 melt and this fluid reference: it illustrates the tracer,
and is not a consistently calibrated THERMOCALC basalt prediction. For
quantitative work supply a consistent model set and applicable calibrations.
See the original [HPx model-family guidance](https://hpxeosandthermocalc.org/the-hpx-eos/using-the-hpx-eos/),
[Pitzer–Sterner EOS](https://doi.org/10.1063/1.467624) and
[NIST water reference](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=37&Units=SI).

The pyrolite example uses the Stixrude–Lithgow-Bertelloni **2024** dataset,
including ferric endmembers, high/low spin states, and bcc, fcc and hcp iron:

```sh
python examples/example_pyrolite_pseudosection.py --output-dir pyrolite_output
python examples/example_pyrolite_pseudosection.py --quick --no-plots
python examples/example_pyrolite_pseudosection.py --plot-json pyrolite_output/pyrolite_pseudosection.json --label-fontsize 8
python examples/example_pyrolite_pseudosection.py --refine-json pyrolite_output/pyrolite_pseudosection.json
```

It covers **0–150 GPa and 0–4000 K**, including the analytic zero-temperature
limits. The bulk is the Workman–Hart pyrolite in Table 1 of
[Stixrude & Lithgow-Bertelloni (2024)](https://doi.org/10.1093/gji/ggae126),
with unchanged cation ratios and a starting **molar Fe³⁺/ΣFe of exactly 3%**.
Oxygen remains closed; ferric iron and metal can subsequently form by
ferrous-iron disproportionation. The full ferropericlase and bridgmanite
endmember coordinates equilibrate their spin populations in the same native
solve as the compositions. The catalogue exposes all 15 unrelaxed solution
models and 74 endmembers as `burnman_cpp.minerals.SLB_2024`. It does not expose
separate relaxed-solution wrappers; spin equilibration is already included in
the phase-equilibrium calculation.

This dataset models **solids**, without silicate or iron liquids. The wide
window is therefore a solid-phase extrapolation, not a melting diagram.
The native SLB volume solver follows a mechanically stable, positive-`K_T`
branch and diagnoses its thermal spinodal or finite-strain Debye limit.
`settings.exclude_invalid_eos=True` is the standard policy: it excludes only
those diagnosed EOS failures; other numerical errors remain failures. Set it
to `False` to require all candidate EOS to be valid. Every affected sample
records `excluded_phases`. If the remaining admissible compositions cannot
represent the bulk, `outside_model_domain=True` distinguishes that limit from
a failed equilibrium solve. Native `result.excluded_regions` records the
low-pressure model boundary, and `plot_pseudosection()` hatches those
regions instead of giving them a phase-count colour. Excluding a solution
with an invalid endmember is conservative; these regions are not a liquid
field or a prediction of a solidus. The pyrolite example sets
`settings.required_eos_phases=['opx']`: its Mg-Tschermak endmember reaches a
thermal spinodal while orthopyroxene is still an essential competitor. The
example therefore conservatively masks that whole-model EOS limit, even
where the remaining candidates could represent the bulk. Nearby phase-line
endpoints keep their accepted thermodynamic states and join the plotted model
boundary within the supplied node tolerance.

`settings.active_solution_faces=True` is also standard. At cold
conditions, the native polytope vertices restrict equilibrium solves to active
zero-site-occupancy faces. Stability checks still search the **full** solution
polytopes, so an incorrectly restricted composition cannot be accepted as
stable. This supplies the inequality conditions needed at zero kelvin, where
forcing all endmembers into interior chemical equilibrium is inappropriate.
Saved compositions retain the original endmember coordinates, including
signed coordinates in nontrivial solution bases, and can be resumed or
independently checked. Continuation releases or restricts these composition
faces when a trace site population changes, using the same native equilibrium
constraints. This avoids mistaking activation of a spin component for the
appearance of another, compositionally identical phase. Set it to `False`
to use interior equilibrium equations throughout; that mode requires T > 0.

The native SLB equations are checked against Python BurnMan in
`python/tests/test_slb_reference.py`: all 74 canonical SLB2024 endmembers,
12 thermodynamic/elastic properties, SLB2/SLB3/conductive variants, independent
Gibbs derivatives and Maxwell relations, thermal spinodals, real-Debye limits,
stishovite softening, and magnetic Gibbs derivatives including the T=0 limit.
The reference tests require Python BurnMan with `SLB_2024` available; this
requires a reference checkout containing the stable-branch and finite-limit
fixes described below. Run them with
`python -m pytest -q python/tests/test_slb_reference.py` in an environment
containing both packages.

The reference volume solver must recover the compressed coesite root at 150 GPa and 0 or
4000 K, confirmed independently by a SciPy Brent solve of the pressure
equation. Both implementations reject the disconnected expanded bcc-iron branch at 4000 K
and 0 or 3 GPa; its reference-connected spinodal is approximately 9.897 GPa.
Both implementations evaluate the finite limits of the magnetic correction at T=0 and the
stishovite shear modulus at its soft-mode pressure. The parity tests
require these fixes in the Python reference checkout.

The magnetic finite-limit evaluation is an exact algebraic rewrite of the
existing CHS Gibbs potential. Its coefficients and reference state are
unchanged. Additional tests prove that identity symbolically and compare
the complete SLB2024 magnetic terms against the authors' HeFESTo `hillert.f`
equations, including the ordered-state offsets and zero-kelvin limit. The
printed SLB2024 equation A5 appears to contain misplaced brackets; the code
follows the authors' implementation. A change to the magnetic model's
functional form or reference state should be a separately named option.
