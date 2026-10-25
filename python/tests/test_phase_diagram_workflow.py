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
"""Small real calculations covering the workflow, persistence and natural axes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024, load_dataset
from burnman_cpp.tools import phase_diagram_workflow as workflow


def calculate(directory, diagram="PT", **kwargs):
    kwargs.setdefault("make_grid", True)
    return bm.calculate_phase_diagram_grid(
        "SLB_2024",
        bm.Composition({"MgO": 2.0}, "molar"),
        diagram,
        {"P": 1e9, "T": 1000.0},
        {"P": 2e9, "T": 1100.0},
        seeds=(2, 2),
        boundary_resolution=(4, 5),
        grid_resolution=(3, 4),
        phase_names=["periclase"],
        chunk_size=3,
        output_dir=directory,
        **kwargs,
    )


def process(run, **kwargs):
    return bm.calculate_potential_grid(
        run.grid_manifest,
        run.phases,
        output_dir=run.refined_diagram.parent / "potentials",
        **kwargs,
    )


@pytest.mark.parametrize(
    "diagram,symbol",
    [
        ("PT", "G"),
        ("TP", "G"),
        ("PS", "H"),
        ("SP", "H"),
        ("VT", "F"),
        ("TV", "F"),
        ("VS", "E"),
        ("SV", "E"),
    ],
)
def test_real_diagram_refinement_saved_grid_and_potential_replay(
    tmp_path, diagram, symbol
):
    run = calculate(tmp_path, diagram)
    potentials = process(run)
    assert run.diagram.resolved and run.diagram.spacing_resolved
    assert run.grid_manifest is not None and not run.diagnostics
    assert run.initial_diagram.exists() and run.refined_diagram.exists()
    manifest = json.loads(run.grid_manifest.read_text())
    assert manifest["complete"] and manifest["equilibrium_successes"] == 12
    grids = bm.load_potential_grid(run.grid_manifest, run.phases)
    assert grids.potential == symbol
    assert grids.values.shape == (
        (4, 3) if tuple(diagram) == grids.variables else (3, 4)
    )
    saved = bm.load_potential_grid(potentials.raw_potential)
    assert saved.potential == symbol
    np.testing.assert_array_equal(saved.values, grids.values)
    assert grids.equilibrium_success.all() and np.isfinite(grids.values).all()
    assert np.isfinite(grids.d2_dxdy).all()
    # Check gradients/Hessians against independently evaluated single-phase EOS.
    phase = SLB_2024.periclase()
    for rows, cols, batch in bm.load_property_grid_chunks(run.grid_manifest):
        coordinates = bm.evaluate_property_grid(
            batch, run.phases, ["pressure", "temperature"]
        ).values
        for j, i in np.ndindex(batch.equilibrium_success.shape):
            solution = batch.solution_at(j, i)
            pressure, temperature = (
                coordinates["pressure"][j, i],
                coordinates["temperature"][j, i],
            )
            phase.set_state(pressure, temperature)
            n = solution.n_moles
            # One mineral formula contains 4 MgO units; n follows that basis.
            volume, cp, cv = (
                n * phase.molar_volume,
                n * phase.molar_heat_capacity_p,
                n * phase.molar_heat_capacity_v,
            )
            beta_t, beta_s = (
                phase.isothermal_compressibility_reuss,
                phase.isentropic_compressibility_reuss,
            )
            alpha = phase.thermal_expansivity
            expected = {
                "G": (
                    volume,
                    -n * phase.molar_entropy,
                    -volume * beta_t,
                    volume * alpha,
                    -cp / temperature,
                ),
                "H": (
                    volume,
                    temperature,
                    -volume * beta_s,
                    alpha * volume * temperature / cp,
                    temperature / cp,
                ),
                "F": (
                    -pressure,
                    -n * phase.molar_entropy,
                    1 / (volume * beta_t),
                    -alpha / beta_t,
                    -cv / temperature,
                ),
                "E": (
                    -pressure,
                    temperature,
                    1 / (volume * beta_s),
                    -alpha * temperature / (beta_t * cv),
                    temperature / cv,
                ),
            }[symbol]
            row, col = rows.start + j, cols.start + i
            if run.diagram.diagram_type in ("TV", "SV"):
                row, col = col, row
            observed = [
                getattr(grids, name)[row, col]
                for name in ("d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2")
            ]
            np.testing.assert_allclose(observed, expected, rtol=2e-6)
            assert grids.values[row, col] == pytest.approx(
                n
                * getattr(
                    phase,
                    {
                        "G": "molar_gibbs",
                        "H": "molar_enthalpy",
                        "F": "molar_helmholtz",
                        "E": "molar_internal_energy",
                    }[symbol],
                )
            )
    # Replay changes neither solve counts nor files on disk.
    assert manifest == json.loads(run.grid_manifest.read_text())


@pytest.mark.parametrize("problem", ["unresolved", "polygon"])
def test_only_completely_resolved_diagrams_are_gridded(tmp_path, monkeypatch, problem):
    refine = workflow._core.refine_phase_diagram

    def fail_refinement(*args, **kwargs):
        result = refine(*args, **kwargs)
        data = result.to_dict()
        if problem == "unresolved":
            data.update(resolved=False, diagnostics=["Unresolved test boundary"])
        return bm.PhaseDiagramResult.from_dict(data)

    monkeypatch.setattr(workflow._core, "refine_phase_diagram", fail_refinement)
    if problem == "polygon":
        monkeypatch.setattr(
            workflow._core,
            "phase_diagram_field_polygons",
            lambda _: SimpleNamespace(diagnostics=["Provisional test field"]),
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Unresolved diagrams must never start gridding")

    monkeypatch.setattr(workflow, "phase_diagram_grid_chunks", forbidden)
    run = calculate(tmp_path)
    assert run.grid_manifest is None and run.diagnostics
    assert run.initial_diagram.exists() and run.refined_diagram.exists()
    assert not (tmp_path / "grid_solutions").exists()
    assert not list(tmp_path.glob("*.npz"))
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["gridding_started"]


@pytest.mark.parametrize("grid_warning", [False, True])
def test_unmet_boundary_spacing_warns_without_blocking_gridding(
    tmp_path, monkeypatch, grid_warning
):
    refine = workflow._core.refine_phase_diagram
    retry = workflow.retry_property_grid_chunks

    def unmet_spacing(*args, **kwargs):
        data = refine(*args, **kwargs).to_dict()
        data.update(spacing_resolved=False, spacing_diagnostics=["Unmet test spacing"])
        return bm.PhaseDiagramResult.from_dict(data)

    monkeypatch.setattr(workflow._core, "refine_phase_diagram", unmet_spacing)
    if grid_warning:

        def warning_on_retry(*args, **kwargs):
            for rows, cols, grid in retry(*args, **kwargs):
                data = grid.to_dict()
                data["diagnostics"] = ["Test grid property warning"]
                yield rows, cols, bm.PhaseDiagramPropertyGrid.from_dict(data)

        monkeypatch.setattr(workflow, "retry_property_grid_chunks", warning_on_retry)
    run = calculate(tmp_path)
    assert run.diagram.resolved and not run.diagram.spacing_resolved
    assert run.grid_manifest is not None
    manifest = json.loads(run.grid_manifest.read_text())
    assert manifest["complete"] and manifest["equilibrium_successes"] == 12
    assert run.diagnostics == [*manifest["diagnostics"], "Unmet test spacing"]
    if grid_warning:
        assert manifest["diagnostics"]
        assert all(
            item["messages"] == ["Test grid property warning"]
            for item in manifest["diagnostics"]
        )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gridding_started"] and report["diagnostics"] == run.diagnostics


@pytest.mark.parametrize("dataset", ["SLB_2024", SLB_2024])
def test_catalogue_aliases_and_endmembers_are_not_duplicated(dataset):
    phases, provenance = load_dataset(dataset)
    assert phases and len({p.name for p in phases}) == len(phases)
    assert {p.name for p in phases} <= SLB_2024.phase_aliases.keys()
    assert len(provenance["phase_names"]) == len(phases)
    assert [getattr(SLB_2024, name)().name for name in provenance["phase_names"]] == [
        phase.name for phase in phases
    ]
    assert not any(isinstance(p, bm.RelaxedSolution) for p in phases)
    endmembers = {
        name for p in phases if isinstance(p, bm.Solution) for name in p.endmember_names
    }
    assert not any(
        p.name in endmembers for p in phases if not isinstance(p, bm.Solution)
    )


def test_refines_a_real_phase_boundary_before_gridding(tmp_path):
    settings = bm.PhaseDiagramSettings()
    settings.step = 0.2
    settings.minimization_starts = 2
    run = bm.calculate_phase_diagram_grid(
        "SLB_2024",
        {"Si": 1.0, "O": 2.0},
        "PT",
        {"P": 0.5e9, "T": 900.0},
        {"P": 5e9, "T": 1500.0},
        phase_names=["quartz", "coesite"],
        settings=settings,
        seeds=(3, 3),
        boundary_resolution=(41, 41),
        grid_resolution=(3, 4),
        output_dir=tmp_path,
        make_grid=True,
    )
    assert run.diagram.resolved and run.diagram.spacing_resolved
    before = bm.load_phase_diagram(run.initial_diagram)
    assert len(run.diagram.boundaries) == 1
    assert sum(len(b.points) for b in run.diagram.boundaries) > sum(
        len(b["points"]) for b in before["boundaries"]
    )
    assert run.grid_manifest is not None
    grids = bm.load_potential_grid(run.grid_manifest, run.phases)
    assert len(np.unique(grids.field_numbers)) == 2
    assert grids.equilibrium_success.all()


def test_requested_order_bounds_and_settings_are_preserved(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "No corner solves are needed when its coordinates match the diagram"
        )

    monkeypatch.setattr(workflow._core, "stable_equilibrium", forbidden)
    settings = bm.PhaseDiagramSettings()
    before = settings.to_dict()
    run = bm.calculate_phase_diagram_grid(
        SLB_2024,
        {"Mg": 1.0, "O": 1.0},
        "T-P",
        {"P": 1e9, "T": 1000.0},
        {"P": 2e9, "T": 1100.0},
        phase_names=["periclase"],
        settings=settings,
        seeds=(3, 2),
        boundary_resolution=(5, 4),
        grid_resolution=(4, 3),
        pressure_range=(1.1e9, 1.9e9),
        output_dir=tmp_path,
        make_grid=True,
    )
    assert settings.to_dict() == before
    assert run.diagram.settings.temperature_seeds == 3
    assert run.diagram.settings.pressure_seeds == 2
    assert run.diagram.pressure_range == pytest.approx((1.1e9, 1.9e9))
    manifest = json.loads(run.grid_manifest.read_text())
    assert manifest["shape"] == [4, 3] and manifest["diagram_type"] == "PT"
    with pytest.raises(FileExistsError):
        calculate(tmp_path)


def test_invalid_requests_do_not_create_output(tmp_path):
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="grid_resolution"):
        bm.calculate_phase_diagram_grid(
            "SLB_2024",
            {"Mg": 1, "O": 1},
            "PT",
            {"P": 1e9, "T": 1000},
            {"P": 2e9, "T": 1200},
            grid_resolution=(3.5, 4),
            output_dir=directory,
        )
    assert not directory.exists()


def test_zero_kelvin_corner_is_accepted_without_coordinate_conversion():
    assert workflow._corner({"P": 0.0, "T": 0.0}, "PT", {}, [], None, {}) == {
        "P": 0.0,
        "T": 0.0,
    }
    with pytest.raises(ValueError, match="nonnegative"):
        workflow._corner({"P": 0.0, "T": -1.0}, "PT", {}, [], None, {})


def test_gridding_can_be_disabled_for_a_resolved_diagram(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Gridding was explicitly disabled")

    monkeypatch.setattr(workflow, "phase_diagram_grid_chunks", forbidden)
    run = calculate(tmp_path, make_grid=False)
    assert run.diagram.resolved and not run.diagnostics
    assert run.grid_manifest is None
    assert run.initial_diagram.exists() and run.refined_diagram.exists()
    assert not (tmp_path / "grid_solutions").exists()


def test_raw_potential_output_does_not_implicitly_process_arrays(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Infilling/smoothing was not requested")

    monkeypatch.setattr(workflow, "infill_potential_grid", forbidden)
    monkeypatch.setattr(workflow, "smooth_potential_grid", forbidden)
    run = calculate(tmp_path, "PS")
    potentials = process(run, smoothing_sigma=0.5)
    raw = bm.load_potential_grid(potentials.raw_potential)
    expected = bm.load_potential_grid(run.grid_manifest, run.phases)
    assert (
        potentials.infilled_potential is None and potentials.smoothed_potential is None
    )
    np.testing.assert_array_equal(raw.values, expected.values)
    np.testing.assert_array_equal(raw.d2_dy2, expected.d2_dy2)


def test_infill_output_does_not_require_smoothing(tmp_path):
    potentials = process(calculate(tmp_path), make_infilled_potential_grid=True)
    assert potentials.raw_potential.exists() and potentials.infilled_potential.exists()
    assert potentials.smoothed_potential is None


def test_smoothing_is_skipped_when_infilling_leaves_a_gap(tmp_path, monkeypatch):
    from copy import copy

    def incomplete(grids, **kwargs):
        result = copy(grids)
        result.values = grids.values.copy()
        result.values[0, 0] = np.nan
        return result, {"success": False}

    monkeypatch.setattr(workflow, "infill_potential_grid", incomplete)
    potentials = process(
        calculate(tmp_path),
        make_infilled_potential_grid=True,
        make_smoothed_potential_grid=True,
    )
    assert potentials.raw_potential.exists() and potentials.infilled_potential.exists()
    assert potentials.smoothed_potential is None
    assert any("nonfinite" in message for message in potentials.diagnostics)


def test_infill_and_smoothing_write_separate_files_and_preserve_raw_gaps(
    tmp_path, monkeypatch
):
    replay = workflow.load_potential_grid

    def missing_corner(*args, **kwargs):
        grids = replay(*args, **kwargs)
        for name in ("values", "d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2"):
            getattr(grids, name)[0, 0] = np.nan
        grids.equilibrium_success[0, 0] = False
        return grids

    monkeypatch.setattr(workflow, "load_potential_grid", missing_corner)
    # Smoothing infers G for PT and requests infilling first.
    potentials = process(
        calculate(tmp_path),
        make_infilled_potential_grid=True,
        make_smoothed_potential_grid=True,
    )
    raw, filled, smoothed = [
        bm.load_potential_grid(path)
        for path in (
            potentials.raw_potential,
            potentials.infilled_potential,
            potentials.smoothed_potential,
        )
    ]
    assert np.isnan(raw.values[0, 0])
    assert np.isfinite(filled.values).all() and np.isfinite(smoothed.values).all()
    observed = np.isfinite(raw.values)
    np.testing.assert_array_equal(filled.values[observed], raw.values[observed])
    np.testing.assert_array_equal(smoothed.equilibrium_success, raw.equilibrium_success)
    assert not smoothed.equilibrium_success[0, 0]
    assert (
        len(
            {
                potentials.raw_potential,
                potentials.infilled_potential,
                potentials.smoothed_potential,
            }
        )
        == 3
    )
    report = json.loads((potentials.raw_potential.parent / "report.json").read_text())
    assert report["postprocessing"]["n_missing"] == 1


@pytest.mark.parametrize(
    "options",
    [
        dict(make_smoothed_potential_grid=True),
        dict(
            make_infilled_potential_grid=True,
            make_smoothed_potential_grid=True,
            smoothing_sigma=-1.0,
        ),
        dict(make_infilled_potential_grid="yes"),
        dict(
            make_infilled_potential_grid=True,
            make_smoothed_potential_grid=True,
            smoothing_sigma=(1, 2, 3),
        ),
    ],
)
def test_invalid_processing_options_are_rejected_before_loading(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        bm.calculate_potential_grid(
            "missing.npz", output_dir=tmp_path / "invalid", **options
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source_type", ["npz", "arrays"])
def test_potential_processing_can_repeat_without_models_or_equilibrium_solves(
    tmp_path, monkeypatch, source_type
):
    run = calculate(tmp_path / "diagram", "PS")
    first = process(run)
    source = first.raw_potential
    before = source.read_bytes()
    if source_type == "arrays":
        source = bm.load_potential_grid(source)

    def forbidden(*args, **kwargs):
        pytest.fail("Processing existing potential arrays must not solve equilibria")

    monkeypatch.setattr(workflow._core, "stable_equilibrium", forbidden)
    monkeypatch.setattr(workflow, "phase_diagram_grid_chunks", forbidden)
    second = bm.calculate_potential_grid(
        source,
        output_dir=tmp_path / "reprocessed",
        make_infilled_potential_grid=True,
        make_smoothed_potential_grid=True,
        smoothing_sigma=(1.0, 2.0),
    )
    assert second.infilled_potential.exists() and second.smoothed_potential.exists()
    assert not second.diagnostics
    assert set(second.timings) == {"potential", "infill", "smoothing"}
    np.testing.assert_array_equal(
        bm.load_potential_grid(second.raw_potential).values,
        bm.load_potential_grid(first.raw_potential).values,
    )
    assert first.raw_potential.read_bytes() == before
    with pytest.raises(FileExistsError):
        bm.calculate_potential_grid(source, output_dir=tmp_path / "reprocessed")
    assert first.raw_potential.read_bytes() == before


def test_smoothing_grid_size_is_checked_before_creating_output(tmp_path):
    grid = bm.PotentialGrid("H", [1.0, 2.0], [3.0, 4.0], np.zeros((2, 2)))
    directory = tmp_path / "potential"
    with pytest.raises(ValueError, match="three grid points"):
        bm.calculate_potential_grid(
            grid,
            output_dir=directory,
            make_infilled_potential_grid=True,
            make_smoothed_potential_grid=True,
        )
    assert not directory.exists()


def test_make_grid_saves_only_equilibrium_samples(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Potential processing was not requested")

    monkeypatch.setattr(workflow, "load_potential_grid", forbidden)
    run = calculate(tmp_path)
    assert run.grid_manifest.exists()
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.parametrize("diagram,upper_temperature", [("PV", 1100.0), ("TS", 1000.01)])
def test_non_natural_sections_can_grid_equilibrium_samples(
    tmp_path, diagram, upper_temperature
):
    run = bm.calculate_phase_diagram_grid(
        "SLB_2024",
        {"Mg": 1.0, "O": 1.0},
        diagram,
        {"P": 1e9, "T": 1000.0},
        {"P": 2e9, "T": upper_temperature},
        seeds=(2, 2),
        boundary_resolution=(3, 3),
        grid_resolution=(3, 4),
        phase_names=["periclase"],
        output_dir=tmp_path,
        make_grid=True,
    )
    assert run.diagram.resolved, run.diagnostics
    assert run.grid_manifest is not None
    assert (
        sum(
            int(grid.equilibrium_success.sum())
            for _, _, grid in bm.load_property_grid_chunks(run.grid_manifest)
        )
        == 12
    )
    with pytest.raises(ValueError, match="Potential processing requires"):
        process(run)
    assert not (tmp_path / "potentials").exists()


@pytest.mark.parametrize("resolved", [True, False])
def test_plotting_without_gridding_saves_and_closes_the_refined_figure(
    tmp_path, monkeypatch, resolved
):
    plt = pytest.importorskip("matplotlib.pyplot")
    refine = workflow._core.refine_phase_diagram
    if not resolved:

        def unresolved(*args, **kwargs):
            data = refine(*args, **kwargs).to_dict()
            data.update(resolved=False, diagnostics=["Test unresolved boundary"])
            return bm.PhaseDiagramResult.from_dict(data)

        monkeypatch.setattr(workflow._core, "refine_phase_diagram", unresolved)
    existing_figures = set(plt.get_fignums())
    run = calculate(
        tmp_path, make_grid=False, plot=True, plot_kwargs={"pressure_unit": "GPa"}
    )
    assert run.grid_manifest is None
    assert run.diagram_plot == tmp_path / "phase_diagram.png"
    assert run.diagram_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert set(plt.get_fignums()) == existing_figures
    assert "plot" in run.timings


@pytest.mark.parametrize(
    "formats,extensions",
    [
        ("pdf", ("pdf",)),
        (["png", "svg"], ("png", "svg")),
        ([".SVG", "svg", ".png"], ("svg", "png")),
    ],
)
def test_plot_formats_accept_string_or_list_and_save_each_format(
    tmp_path, formats, extensions
):
    pytest.importorskip("matplotlib")
    run = calculate(tmp_path, make_grid=False, plot=True, plot_formats=formats)
    expected = tuple(
        tmp_path / f"phase_diagram.{extension}" for extension in extensions
    )
    assert run.diagram_plots == expected
    assert run.diagram_plot == expected[0]
    assert all(path.stat().st_size > 0 for path in expected)
    assert set(tmp_path.glob("phase_diagram.*")) == set(expected)


@pytest.mark.parametrize("formats", [[], ["not-a-format"]])
def test_bad_plot_formats_are_rejected_before_calculation(tmp_path, formats):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="format"):
        calculate(tmp_path, plot=True, plot_formats=formats)
    assert not list(tmp_path.iterdir())
