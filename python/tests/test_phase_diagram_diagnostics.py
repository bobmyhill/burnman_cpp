# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Diagnostic ownership and provenance survive wording changes and persistence."""

from copy import deepcopy

import pytest
import burnman_cpp as bm
from conftest import diagnostic


@pytest.fixture(scope="module")
def single_phase():
    phase = bm.minerals.HGP_2018_ds633.per()
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = 2
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], [1.0e9, 2.0e9], [800.0, 1000.0], opts
    )
    return result.to_dict(), phase


def test_diagnostic_records_roundtrip_with_readable_views(single_phase, tmp_path):
    data = deepcopy(single_phase[0])
    records = [
        diagnostic("a saved connection failed", "trace", 7),
        diagnostic("custom grid wording", "equilibrium", grid_point=[2, 3]),
        diagnostic("a user note"),
    ]
    data.update(resolved=False, diagnostics=records)
    result = bm.PhaseDiagramResult.from_dict(data)
    assert result.diagnostics == [
        "Boundary 7: a saved connection failed",
        "grid [2, 3]: custom grid wording",
        "a user note",
    ]
    assert result.diagnostic_records == records
    path = bm.save_phase_diagram(result, tmp_path / "records.json")
    restored = bm.load_phase_diagram(path)
    assert restored["schema_version"] == 6
    assert restored["diagnostics"] == records
    assert bm.PhaseDiagramResult.from_dict(restored).diagnostic_records == records


def test_refresh_uses_categories_even_when_wording_changes(single_phase):
    data, phase = deepcopy(single_phase[0]), single_phase[1]
    kept = [
        diagnostic(
            "Field 999 has conflicting perimeter reactions; its label is provisional."
        ),
        diagnostic("Thermodynamic audit: an unrelated user note"),
        diagnostic("Untraced branches remain", "search"),
    ]
    owned = [
        diagnostic("completely revised wording", category, 999)
        for category in (
            "field",
            "boundary",
            "junction",
            "critical_endpoint",
            "missing_assemblage",
            "thermodynamic",
            "sample",
            "trace",
        )
    ]
    data.update(resolved=False, diagnostics=owned + kept)
    note = diagnostic("Unrelated coordinate information")
    data["coordinate_diagnostics"] = [note]
    result = bm.verify_phase_diagram_field_perimeters(data, [phase])
    assert result.diagnostic_records == kept
    assert result.coordinate_diagnostic_records == [note]
    assert not result.resolved


@pytest.mark.parametrize(
    "record, error",
    [
        ("old text-only warning", "category/message"),
        (
            dict(category="unknown_kind", message="problem"),
            "Unknown diagnostic category",
        ),
        (diagnostic("problem", "trace", -1), "entity_id"),
        (diagnostic("problem", "equilibrium", grid_point=[0, -1]), "grid_point"),
    ],
)
def test_invalid_diagnostic_records_are_rejected(single_phase, record, error):
    data = dict(single_phase[0], diagnostics=[record])
    with pytest.raises(ValueError, match=error):
        bm.PhaseDiagramResult.from_dict(data)
