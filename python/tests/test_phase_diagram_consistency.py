# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Recover a saved metastable composition branch with an unchanged phase set."""

import copy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm


@pytest.fixture
def retained_stencil():
    root = Path(__file__).parents[2]
    record = json.loads(
        (Path(__file__).parent / "data/pyrolite_sp_enthalpy_branch.json").read_text()
    )
    models = runpy.run_path(str(root / "examples/example_pyrolite_phase_diagram.py"))
    return record, models["candidate_phases"]


def check_repaired(grid, record):
    assert grid.resolved, grid.diagnostics
    assert grid.equilibrium_success.all()
    assert grid.values["temperature"][1, 0] == pytest.approx(
        record["expected_middle_temperature_K"], abs=0.01
    )
    solution = grid.solution_at(1, 0)
    enthalpy = solution.gibbs + solution.state.temperature * solution.state.entropy
    assert enthalpy == pytest.approx(record["expected_middle_enthalpy_J"], abs=0.01)
    # Both endpoints are already stable, so the repaired enthalpies must form
    # a convex stencil and temperatures must increase with entropy.
    solutions = [grid.solution_at(j, 0) for j in range(3)]
    enthalpies = [s.gibbs + s.state.temperature * s.state.entropy for s in solutions]
    assert enthalpies[0] - 2 * enthalpies[1] + enthalpies[2] >= 0.0
    assert (np.diff(grid.values["temperature"][:, 0]) >= 0.0).all()


def test_reverse_checks_successful_roots_and_preserves_stable_endpoints(
    retained_stencil,
):
    record, factory = retained_stencil
    original = copy.deepcopy(record["grid"])
    grid = bm.PhaseDiagramPropertyGrid.from_dict(original)
    prepared = bm._core._PropertyGridder(record["source"], factory(), grid.x, grid.y)
    prepared.retry(grid)
    check_repaired(grid, record)
    assert grid.equilibrium_solves > original["equilibrium_solves"]
    assert record["grid"] == original
    # The phase names cannot distinguish the correct and wrong middle roots.
    assert sorted(p.name for p in grid.solution_at(1, 0).state.phases) == sorted(
        p["name"] for p in original["solutions"][1][0]["phases"]
    )
    for j in (0, 2):
        assert (
            grid.values["temperature"][j, 0] == original["values"]["temperature"][j][0]
        )
    solves = grid.equilibrium_solves
    prepared.retry(grid)
    assert grid.equilibrium_solves == solves


def test_chunk_retry_saves_repairs_to_previously_successful_points(
    retained_stencil, tmp_path
):
    record, factory = retained_stencil
    chunks = []
    for j in range(3):
        data = copy.deepcopy(record["grid"])
        data["y"] = data["y"][j : j + 1]
        for key in ("solutions", "field_numbers", "success", "equilibrium_success"):
            data[key] = data[key][j : j + 1]
        data["values"] = {k: v[j : j + 1] for k, v in data["values"].items()}
        filename = f"chunk_{j}.json"
        bm.save_property_grid(
            bm.PhaseDiagramPropertyGrid.from_dict(data), tmp_path / filename
        )
        chunks.append(dict(file=filename, rows=[j, j + 1], columns=[0, 1]))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="burnman_cpp.property_grid_chunks",
                schema_version=1,
                x=record["grid"]["x"],
                y=record["grid"]["y"],
                chunks=chunks,
            )
        )
    )
    for _ in bm.retry_property_grid_chunks(record["source"], factory(), tmp_path):
        pass
    saved = bm.load_property_grid(tmp_path / "chunk_1.json")
    assert saved.values["temperature"][0, 0] == pytest.approx(
        record["expected_middle_temperature_K"], abs=0.01
    )
    assert saved.equilibrium_solves > 0
