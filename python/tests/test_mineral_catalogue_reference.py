"""Keep catalogue regeneration reproducible against the pinned Python BurnMan."""

import importlib
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pytest

pytest.importorskip("burnman")
import cvxpy as cp
from burnman.minerals import SLB_2011 as pySLB, JH_2015 as pyJH
from burnman_cpp.minerals import SLB_2011 as SLB, JH_2015 as JH

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_FILES = (
    "src/minerals/datasets.cpp",
    "include/burnman/minerals/datasets.hpp",
    "python/bindings/minerals.cpp",
)


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return importlib.import_module("export_example_minerals")


@pytest.mark.parametrize("solver", ["CLARABEL", "HIGHS"])
def test_catalogue_check_is_independent_of_optimizer_choice(
    exporter, monkeypatch, capsys, solver
):
    if shutil.which("clang-format") is None:
        pytest.skip("Catalogue text audit requires clang-format.")
    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed.")
    original_solve = cp.Problem.solve

    def solve(*args, **kwargs):
        kwargs["solver"] = solver
        return original_solve(*args, **kwargs)

    # Both choices changed the legacy polytope-based catalogue: CLARABEL
    # exchanged two orthopyroxene rows, HIGHS selected another valid basis.
    monkeypatch.setattr(cp.Problem, "solve", solve)
    monkeypatch.setattr(sys, "argv", ["export_example_minerals.py", "--check"])
    before = [(ROOT / filename).read_bytes() for filename in CATALOGUE_FILES]
    exporter.main()
    assert "Verified 253 endmembers and 155 factories" in capsys.readouterr().out
    assert [(ROOT / filename).read_bytes() for filename in CATALOGUE_FILES] == before


def test_catalogue_check_rejects_real_data_changes_with_a_diff(
    exporter, monkeypatch, tmp_path, capsys
):
    if shutil.which("clang-format") is None:
        pytest.skip("Catalogue text audit requires clang-format.")
    args = ["export_example_minerals.py", "--output-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    exporter.main()
    path = tmp_path / CATALOGUE_FILES[0]
    changed, count = re.subn(
        r"(m\.params\.H_0 = )([-+\d.]+)",
        lambda match: match[1] + repr(float(match[2]) + 10.0),
        path.read_text(),
        count=1,
    )
    assert count == 1
    path.write_text(changed)
    monkeypatch.setattr(sys, "argv", args + ["--check"])
    with pytest.raises(
        SystemExit, match="Catalogue differs.*src/minerals/datasets.cpp"
    ):
        exporter.main()
    error = capsys.readouterr().err
    assert "--- src/minerals/datasets.cpp" in error
    assert "+++ src/minerals/datasets.cpp (pinned Python BurnMan)" in error
    assert "H_0" in error
    assert path.read_text() == changed


@pytest.mark.parametrize(
    "parent_factory,native_factory,basis,elements",
    [
        (
            pySLB.garnet,
            SLB.pyrope_grossular,
            [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]],
            {"Mg", "Ca", "Al", "Si", "O"},
        ),
        (
            pySLB.mg_fe_bridgmanite,
            SLB.mg_fe_bridgmanite_binary,
            [[1, 0, 0], [0, 1, 0]],
            {"Mg", "Fe", "Si", "O"},
        ),
        (
            pyJH.orthopyroxene,
            JH.mg_fe_orthopyroxene,
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
            ],
            {"Mg", "Fe", "Si", "O"},
        ),
    ],
    ids=["pyrope-grossular", "mg-fe-bridgmanite", "mg-fe-orthopyroxene"],
)
@pytest.mark.parametrize("pressure,temperature", [(1.0e5, 800.0), (3.0e9, 1200.0)])
def test_fixed_reduced_models_preserve_parent_and_native_thermodynamics(
    exporter, parent_factory, native_factory, basis, elements, pressure, temperature
):
    parent = parent_factory()
    reduced = exporter.reduced(parent, basis)
    native = native_factory()
    basis = np.asarray(basis)
    assert reduced.endmember_names == native.endmember_names
    assert set(reduced.elements) == set(native.elements) == elements
    for fractions in [
        np.arange(1, len(basis) + 1, dtype=float),
        np.arange(len(basis), 0, -1, dtype=float),
    ]:
        fractions /= fractions.sum()
        parent.set_composition(basis.T @ fractions)
        reduced.set_composition(fractions)
        native.set_composition(fractions)
        for phase in (parent, reduced, native):
            phase.set_state(pressure, temperature)
        # The full parent retains zero amounts for excluded elements.
        for element in parent.elements:
            assert reduced.formula.get(element, 0.0) == pytest.approx(
                parent.formula.get(element, 0.0), abs=1.0e-14
            )
            assert native.formula.get(element, 0.0) == pytest.approx(
                parent.formula.get(element, 0.0), abs=1.0e-14
            )
        for prop in ("molar_gibbs", "molar_entropy", "molar_volume"):
            assert getattr(reduced, prop) == pytest.approx(
                getattr(parent, prop), rel=1.0e-12, abs=1.0e-18
            )
            assert getattr(native, prop) == pytest.approx(
                getattr(reduced, prop), rel=5.0e-8, abs=1.0e-18
            )
