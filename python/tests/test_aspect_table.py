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
"""ASPECT units/order, elastic averaging, missing velocities and rejected tables."""

from dataclasses import replace
import json

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24
from burnman_cpp.tools import aspect


@pytest.fixture(scope="module")
def calculation(tmp_path_factory):
    directory = tmp_path_factory.mktemp("aspect")
    composition = bm.Composition({"MgO": 1.0, "SiO2": 1.0}, "molar")
    run = bm.calculate_phase_diagram_grid(
        "SLB24",
        composition,
        "PS",
        {"P": 25e9, "T": 1000.0},
        {"P": 26e9, "T": 1100.0},
        phase_names=["periclase", "stishovite"],
        seeds=(2, 2),
        boundary_resolution=(3, 3),
        grid_resolution=(5, 4),
        make_grid=True,
        output_dir=directory,
    )
    assert run.grid_manifest is not None, run.diagnostics
    potentials = bm.calculate_potential_grid(
        run.grid_manifest, run.phases, output_dir=directory / "potentials"
    )
    grids = bm.load_potential_grid(potentials.raw_potential)
    batch = next(bm.load_property_grid_chunks(run.grid_manifest))[2]
    assert batch.equilibrium_success.all()
    mass = sum(composition.mass_composition.values())
    assert batch.bulk_composition == composition.atomic_composition
    return run, grids, mass, batch


def analytic_enthalpy(grids, mass):
    """A stable quadratic with a temperature different from the saved states."""
    p, s = np.meshgrid(grids.x, grids.y / mass)
    return replace(
        grids,
        values=mass
        * (-1e6 + 1000 * s + 2.5 * s**2 + 3e-4 * p - 0.5e-15 * p**2 + 1e-9 * p * s),
        d_dx=mass * (3e-4 - 1e-15 * p + 1e-9 * s),
        d_dy=1000 + 5 * s + 1e-9 * p,
        d2_dx2=np.full(p.shape, -mass * 1e-15),
        d2_dxdy=np.full(p.shape, 1e-9),
        d2_dy2=np.full(p.shape, 5 / mass),
    )


@pytest.mark.parametrize("source_kind", ["native", "json", "chunks"])
def test_table_uses_smoothed_thermodynamics_and_native_elastic_states(
    calculation, tmp_path, monkeypatch, source_kind
):
    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    source = {
        "native": saved,
        "json": bm.save_property_grid(saved, tmp_path / "saved.json.gz"),
        "chunks": run.grid_manifest,
    }[source_kind]
    original = saved.to_dict()
    replay = aspect.evaluate_property_grid
    calls = []

    def evaluate(*args, **kwargs):
        result = replay(*args, **kwargs)
        assert result.equilibrium_solves == 0
        assert not kwargs["relaxed"]
        calls.append(kwargs["averaging_scheme"])
        return result

    monkeypatch.setattr(aspect, "evaluate_property_grid", evaluate)
    export = bm.write_aspect_entropy_pressure_table(
        bm.save_potential_grid(grids, tmp_path / "smoothed.npz"),
        source,
        run.phases,
        tmp_path / "table.txt",
    )
    assert export.table is not None and not export.diagnostics
    assert calls and all(scheme == bm.AveragingType.VRH for scheme in calls)
    lines = export.table.read_text().splitlines()
    assert lines[2] == "# POINTS: 4 5"
    assert lines[3].split() == [
        "s,J/K/kg",
        "P(bar)",
        "T(K)",
        "rho,kg/m3",
        "alpha,1/K",
        "cp,J/K/kg",
        "vp,km/s",
        "vs,km/s",
        "h,J/kg",
    ]
    data = np.loadtxt(export.table, skiprows=4)
    assert data.shape == (20, 9)
    for k, row in enumerate(data):
        i, j = divmod(k, len(grids.y))
        p, s = grids.x[i], grids.y[j] / mass
        v = 3e-4 - 1e-15 * p + 1e-9 * s
        t = 1000 + 5 * s + 1e-9 * p
        np.testing.assert_allclose(
            row[:6], [s, p / 1e5, t, 1 / v, 1e-9 / (5 * v), t / 5]
        )
        assert row[8] == pytest.approx(grids.values[j, i] / mass)
        solution = saved.solution_at(j, i)
        reference = bm.Assemblage(
            [SLB24.periclase(), SLB24.stishovite()], solution.molar_fractions
        )
        reference.set_state(solution.state.pressure, solution.state.temperature)
        np.testing.assert_allclose(
            row[6:8],
            [reference.p_wave_velocity / 1e3, reference.shear_wave_velocity / 1e3],
        )
        assert t != pytest.approx(solution.state.temperature)
    assert saved.to_dict() == original


@pytest.mark.parametrize("scheme", list(bm.AveragingType.__members__.values()))
def test_averaging_scheme_matches_native_assemblage(calculation, tmp_path, scheme):
    run, raw, mass, saved = calculation
    original = saved.to_dict()
    names = ["p_wave_velocity", "shear_wave_velocity"]
    result = bm.evaluate_property_grid(
        saved, run.phases, names, relaxed=False, averaging_scheme=scheme
    )
    assert result.equilibrium_solves == 0 and result.success.all(), result.diagnostics
    solution = saved.solution_at(0, 0)
    reference = bm.Assemblage(
        [SLB24.periclase(), SLB24.stishovite()], solution.molar_fractions
    )
    reference.set_averaging_scheme(scheme)
    reference.set_state(solution.state.pressure, solution.state.temperature)
    np.testing.assert_allclose(
        [result.values[name][0, 0] for name in names],
        [reference.p_wave_velocity, reference.shear_wave_velocity],
        rtol=1e-12,
    )
    export = bm.write_aspect_entropy_pressure_table(
        analytic_enthalpy(raw, mass),
        saved,
        run.phases,
        tmp_path / "table.txt",
        averaging_scheme=scheme,
    )
    row = np.loadtxt(export.table, skiprows=4)[0]
    np.testing.assert_allclose(
        row[6:8], [reference.p_wave_velocity / 1e3, reference.shear_wave_velocity / 1e3]
    )
    assert saved.to_dict() == original


def test_voigt_and_reuss_produce_different_velocities(calculation):
    run, _, _, saved = calculation
    velocities = [
        bm.evaluate_property_grid(
            saved, run.phases, "p_wave_velocity", relaxed=False, averaging_scheme=scheme
        ).values["p_wave_velocity"]
        for scheme in (bm.AveragingType.Voigt, bm.AveragingType.Reuss)
    ]
    assert (velocities[0] > velocities[1]).all()


def test_missing_native_solutions_interpolate_only_missing_velocities(
    calculation, tmp_path
):
    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    full = bm.write_aspect_entropy_pressure_table(
        grids, saved, run.phases, tmp_path / "full.txt"
    )
    data = saved.to_dict()
    for j, i in ((0, 0), (1, 2)):
        data["solutions"][j][i] = None
        data["equilibrium_success"][j][i] = False
        data["success"][j][i] = False
    missing = bm.PhaseDiagramPropertyGrid.from_dict(data)
    export = bm.write_aspect_entropy_pressure_table(
        grids, missing, run.phases, tmp_path / "infilled.txt"
    )
    observed = missing.equilibrium_success.T.ravel()
    truth = np.loadtxt(full.table, skiprows=4)
    filled = np.loadtxt(export.table, skiprows=4)
    # Bulk mass is recomputed after JSON reconstruction. Summation order can
    # change derived specific properties by a few floating-point roundoff units.
    np.testing.assert_allclose(
        filled[observed], truth[observed], rtol=4 * np.finfo(float).eps, atol=0
    )
    assert np.isfinite(filled).all()
    assert len(export.diagnostics) == 2
    assert all("interpolated 2" in message for message in export.diagnostics)


def test_velocity_gap_interpolation_reproduces_plane_and_uses_nearest_at_edge():
    x, y = np.linspace(1e9, 5e9, 5), np.linspace(10, 40, 4)
    p, s = np.meshgrid(x, y)
    truth = 8000 + 1e-7 * p + s
    values = truth.copy()
    values[1, 2] = values[0, 0] = np.nan
    observed = np.isfinite(values)
    edge_count = aspect._interpolate_velocity_gaps(values, x, y)
    assert edge_count == 1
    assert values[1, 2] == pytest.approx(truth[1, 2])
    np.testing.assert_array_equal(values[observed], truth[observed])
    assert np.isfinite(values[0, 0])


def test_specific_properties_do_not_depend_on_total_bulk_scale(calculation, tmp_path):
    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    reference = bm.write_aspect_entropy_pressure_table(
        grids, saved, run.phases, tmp_path / "reference.txt"
    )
    data = saved.to_dict()
    data["context"]["composition_start"] = {
        k: 2 * v for k, v in data["context"]["composition_start"].items()
    }
    data["context"]["entropy_range"] = [2 * v for v in data["context"]["entropy_range"]]
    data["y"] = [2 * v for v in data["y"]]
    for row in data["solutions"]:
        for solution in row:
            for name in ("n_moles", "gibbs", "entropy", "volume"):
                solution[name] *= 2
            for phase in solution["phases"]:
                phase["amount"] *= 2
    scaled = replace(
        grids,
        y=2 * grids.y,
        values=2 * grids.values,
        d_dx=2 * grids.d_dx,
        d2_dx2=2 * grids.d2_dx2,
        d2_dy2=grids.d2_dy2 / 2,
        n_moles=2 * grids.n_moles,
    )
    export = bm.write_aspect_entropy_pressure_table(
        scaled,
        bm.PhaseDiagramPropertyGrid.from_dict(data),
        run.phases,
        tmp_path / "scaled.txt",
    )
    assert export.table is not None and not export.diagnostics
    np.testing.assert_allclose(
        np.loadtxt(export.table, skiprows=4),
        np.loadtxt(reference.table, skiprows=4),
        rtol=1e-13,
    )


def test_missing_all_velocities_rejects_table_but_allows_plots(calculation, tmp_path):
    run, raw, mass, saved = calculation
    data = saved.to_dict()
    for name in ("solutions", "success", "equilibrium_success"):
        value = None if name == "solutions" else False
        data[name] = [[value for _ in saved.x] for _ in saved.y]
    export = bm.write_aspect_entropy_pressure_table(
        analytic_enthalpy(raw, mass),
        bm.PhaseDiagramPropertyGrid.from_dict(data),
        run.phases,
        tmp_path / "missing.txt",
        plot=True,
    )
    assert export.table is None and len(export.plots) == 1
    assert "nonfinite vp" in " ".join(export.diagnostics)
    assert not (tmp_path / "missing.txt").exists()


@pytest.mark.parametrize(
    "derivative,value",
    [
        ("d_dy", 0.0),
        ("d_dx", -1e-6),
        ("d2_dy2", -1.0),
        ("d2_dx2", 1e-20),
        ("d2_dxdy", np.nan),
        ("values", np.inf),
    ],
)
def test_invalid_thermodynamics_reject_table(calculation, tmp_path, derivative, value):
    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    getattr(grids, derivative)[1, 1] = value
    path = tmp_path / "table.txt"
    path.write_text("existing table")
    export = bm.write_aspect_entropy_pressure_table(grids, saved, run.phases, path)
    assert export.table is None
    assert export.diagnostics[0] == "Table rejected: 1 invalid nodes."
    assert path.read_text() == "existing table"


def test_rejected_table_still_plots_every_property(calculation, tmp_path):
    import matplotlib.pyplot as plt

    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    grids.d2_dy2[0, 0] = -1
    before = plt.get_fignums()
    export = bm.write_aspect_entropy_pressure_table(
        grids,
        saved,
        run.phases,
        tmp_path / "bad.txt",
        plot=True,
        plot_formats=["png", "svg"],
    )
    assert export.table is None and not (tmp_path / "bad.txt").exists()
    assert {path.suffix for path in export.plots} == {".png", ".svg"}
    assert all(path.stat().st_size > 1000 for path in export.plots)
    svg = next(path for path in export.plots if path.suffix == ".svg").read_text()
    for label in (
        "Temperature",
        "Density",
        "Thermal expansivity",
        "Specific heat",
        "Vp",
        "Vs",
        "Specific enthalpy",
    ):
        assert label in svg
    assert plt.get_fignums() == before


def test_input_validation_and_incomplete_manifest(calculation, tmp_path):
    run, raw, mass, saved = calculation
    grids = analytic_enthalpy(raw, mass)
    path = tmp_path / "table.txt"
    with pytest.raises(TypeError, match="AveragingType"):
        bm.write_aspect_entropy_pressure_table(
            grids, saved, run.phases, path, averaging_scheme="VRH"
        )
    with pytest.raises(ValueError, match="coordinates"):
        bm.write_aspect_entropy_pressure_table(
            replace(grids, x=grids.x + 1), saved, run.phases, path
        )
    with pytest.raises(ValueError, match=r"H\(P,S\)"):
        bm.write_aspect_entropy_pressure_table(
            replace(grids, potential="G"), saved, run.phases, path
        )
    with pytest.raises(ValueError, match="derivatives"):
        bm.write_aspect_entropy_pressure_table(
            replace(grids, d_dx=None), saved, run.phases, path
        )
    manifest = json.loads(run.grid_manifest.read_text())
    manifest["complete"] = False
    incomplete = tmp_path / "manifest.json"
    incomplete.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not finished"):
        bm.write_aspect_entropy_pressure_table(grids, incomplete, run.phases, path)
    assert not path.exists()
