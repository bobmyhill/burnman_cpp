# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""A grid warm start must not freeze sites that become populated on heating."""

import numpy as np
import burnman_cpp as bm
import json
from pathlib import Path
from burnman_cpp.minerals import load_dataset

from test_property_grid import exchange_phases, equilibrated


def test_grid_releases_cold_solution_faces_on_heating():
    bulk = {"Mg": 0.5, "Fe": 0.5, "O": 1.0}
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    phases = exchange_phases()
    diagram = bm.phase_diagram(bulk, phases, (3.0e9, 5.0e9), (0.0, 1200.0), settings)
    assert diagram.resolved, diagram.diagnostics
    grid = bm.phase_diagram_grid(
        diagram,
        phases,
        ["molar_gibbs", "molar_entropy"],
        x=[4.0e9],
        y=[0.0, 1200.0],
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    cold, hot = grid.solution_at(0, 0), grid.solution_at(1, 0)
    assert all(b.shape == (1, 2) for b in cold.phase_bases)
    assert all(b.shape == (2, 2) for b in hot.phase_bases)
    reference = equilibrated(4.0e9, 1200.0)
    np.testing.assert_allclose(
        grid.values["molar_gibbs"][1, 0], reference.molar_gibbs, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        grid.values["molar_entropy"][1, 0], reference.molar_entropy, rtol=1.0e-10
    )


def test_basalt_ps_grid_releases_frozen_bridgmanite_and_trace_sites():
    data = json.loads(
        (Path(__file__).parent / "data/basalt_grid_frozen_faces.json").read_text()
    )
    phases, _ = load_dataset("SLB_2024", data["models"]["phase_names"])
    ref = data["reference"]
    grid = bm.phase_diagram_grid(
        data["diagram"],
        phases,
        ["enthalpy", "temperature"],
        x=[ref["pressure_Pa"]],
        y=[ref["entropy_J_K"]],
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    assert abs(grid.values["enthalpy"][0, 0] - ref["enthalpy_J"]) < 1.0e-3
    assert abs(grid.values["temperature"][0, 0] - ref["temperature_K"]) < 1.0e-3
    assert grid.values["enthalpy"][0, 0] < ref["restricted_enthalpy_J"] - 1.0
    solution = grid.solution_at(0, 0)
    bridgmanite = next(
        i for i, p in enumerate(solution.state.phases) if p.candidate_index == 2
    )
    assert solution.phase_bases[bridgmanite].shape[0] > 1


def test_reopening_sites_can_use_a_shorter_feasible_starting_step():
    data = json.loads(
        (Path(__file__).parent / "data/basalt_grid_face_release_step.json").read_text()
    )
    phases, _ = load_dataset("SLB_2024", data["models"]["phase_names"])
    target = data["target"]
    grid = bm.phase_diagram_grid(
        data["diagram"],
        phases,
        ["temperature"],
        x=[target["pressure_Pa"]],
        y=[target["entropy_J_K"]],
        store_solutions=True,
    )
    assert grid.equilibrium_success.all(), grid.diagnostics
    state = grid.solution_at(0, 0).state
    assert abs(state.pressure - target["pressure_Pa"]) < 1.0
    assert abs(state.entropy - target["entropy_J_K"]) < 1.0e-7
    assert abs(state.temperature - 476.24036884) < 1.0e-3
