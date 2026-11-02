# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Shared driver for Supplement 3 of Green et al. (2025), egae079.

Use the separate example_green2025_*.py scripts. Thermodynamics, equilibrium,
boundary tracing and polygon construction use the standard C++ routines.
"""

from collections import Counter
import json
from pathlib import Path
import time

import burnman_cpp as bm
from burnman_cpp.minerals import HPx_ds636, model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    write_phase_diagram,
    save_figure,
)

SOURCE = "https://doi.org/10.1093/petrology/egae079"


def candidate_phases(composition):
    models = model_sets.igneous("G25")
    # Match MAGEMin's igneous phase exclusions when H2O is absent, including
    # the formal dry corners of biotite, cordierite and aqueous fluid.
    hydrous = {"fluid", "hb", "bi", "ms", "ep", "cd"}
    phases = [
        phase
        for phase in models.phases
        if phase.name != "cor"
        and (
            phase.name not in hydrous or composition.atomic_composition.get("H", 0) > 0
        )
    ]
    # The reference igneous catalogue includes stishovite, not corundum.
    stishovite = HPx_ds636.stv()
    stishovite.set_name("stv")
    return phases + [stishovite]


def compare(composition, reference_points, settings):
    """Check assemblages at points read well inside the published fields.

    The PDF distinguishes pl/afs, augite/pigeonite and spl/mgt/cm/usp by
    composition. These are instances of fsp, cpx and sp respectively here.
    Repeated names preserve coexistence across a solvus.
    """
    rows = []
    phases = candidate_phases(composition)
    for temperature_c, pressure_kbar, expected in reference_points:
        state = bm.stable_equilibrium(
            composition.atomic_composition,
            phases,
            pressure_kbar * 1.0e8,
            temperature_c + 273.15,
            settings,
        )
        actual = sorted(p.name.split(" #")[0] for p in state.phases if p.amount > 0)
        rows.append(
            dict(
                temperature_C=temperature_c,
                pressure_kbar=pressure_kbar,
                expected=sorted(expected.split()),
                calculated=actual,
                success=state.success,
                match=state.success and Counter(actual) == Counter(expected.split()),
                message=state.message,
                mass_balance_error=state.mass_balance_error,
                minimum_affinity_J_mol=state.minimum_affinity,
            )
        )
        print(
            f"  {temperature_c} C, {pressure_kbar} kbar: {actual}; "
            f"{'PASS' if rows[-1]['match'] else 'DIFFERENCE'}",
            flush=True,
        )
    return rows


def compare_liquidus(data, pressure_kbar, reference_temperature):
    """Intersect saved phase lines with the PDF's liquidus comparison isobar."""
    report = dict(
        pressure_kbar=pressure_kbar,
        reference_temperature_C=reference_temperature,
        reference_reading_tolerance_C=5.0,
        calculated_temperature_C=None,
        difference_C=None,
        match=None,
    )
    pressure = data["pressure_range"]
    temperature = data["temperature_range"]
    if not (
        pressure[0] <= pressure_kbar * 1.0e8 <= pressure[1]
        and temperature[0] <= reference_temperature + 273.15 <= temperature[1]
    ):
        return report | dict(
            status="not_applicable",
            reason="Reference point is outside the calculation window.",
        )
    liquid = {"melt", "fluid"}
    temperatures = []
    for line in data["boundaries"]:
        sides = [
            {data["phase_names"][i].split(" #")[0] for i in line[side]}
            for side in ("side_a", "side_b")
        ]
        # Water-undersaturated melt can coexist with no fluid. Exclude a
        # fluid-in line: a liquidus line must have a solid on its other side.
        if not any("melt" in side and side <= liquid for side in sides) or not (
            set.union(*sides) - liquid
        ):
            continue
        for left, right in zip(line["points"], line["points"][1:]):
            p0, p1 = left["pressure"] / 1.0e8, right["pressure"] / 1.0e8
            if p0 != p1 and min(p0, p1) <= pressure_kbar <= max(p0, p1):
                fraction = (pressure_kbar - p0) / (p1 - p0)
                temperatures.append(
                    left["temperature"]
                    + fraction * (right["temperature"] - left["temperature"])
                    - 273.15
                )
    calculated = max(temperatures) if temperatures else None
    return report | dict(
        calculated_temperature_C=calculated,
        difference_C=None if calculated is None else calculated - reference_temperature,
        match=calculated is not None and abs(calculated - reference_temperature) <= 5.0,
        status=(
            "missing_intersection"
            if calculated is None
            else (
                "match"
                if abs(calculated - reference_temperature) <= 5.0
                else "mismatch"
            )
        ),
        reason=(
            "No liquidus intersection inside the calculation window."
            if calculated is None
            else None
        ),
    )


def plot(data, target, temperature_range, pressure_range, title, fontsize):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    bm.plot_phase_diagram(
        data,
        ax=ax,
        cmap="Blues_r",
        fill_alpha=1.0,
        label_fontsize=fontsize,
        phase_aliases={"melt": "liq", "fluid": "fl"},
        label_key_path=target.with_name(target.stem + "_assemblages.md"),
    )
    ax.set(xlim=temperature_range, ylim=pressure_range, title=title)
    fig.subplots_adjust(bottom=0.15)
    fig.text(
        0.12,
        0.035,
        "G25 + dataset 6.36; closed H and O inventories.\n"
        "fsp: pl/afs; cpx: augite/pigeonite; sp: spl/mgt/cm/usp.\n"
        "A prefix counts coexisting compositions; numbered fields have a separate key.",
        fontsize=fontsize,
    )
    save_figure(fig, target, dpi=180)


def main(
    name,
    title,
    figure,
    oxides,
    temperature_range,
    pressure_range,
    reference_points,
    liquidus,
    *,
    seeds=7,
    check_only=False,
):
    output_dir = Path(__file__).resolve().parent / f"green2025_{name}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Oxide moles, including atomic O; normalize to improve numerical scaling.
    composition = bm.Composition(
        {oxide: amount / sum(oxides.values()) for oxide, amount in oxides.items()},
        unit_type="molar",
    )
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = seeds
    settings.step = 0.015
    settings.max_recovery_passes = 2
    settings.verbose = True
    if not check_only:
        started = time.monotonic()
        result = bm.phase_diagram(
            composition.atomic_composition,
            candidate_phases(composition),
            (max(1.0e5, pressure_range[0] * 1.0e8), pressure_range[1] * 1.0e8),
            tuple(t + 273.15 for t in temperature_range),
            settings,
        )
        data = result.to_dict() | dict(
            source=SOURCE,
            reference_figure=f"S3-{figure}",
            oxide_molar_amounts=oxides,
            model_set=model_sets.igneous("G25").to_dict(),
            calculation_seconds=time.monotonic() - started,
        )
        write_phase_diagram(data, output_dir / name)
        plot(
            data,
            output_dir / f"{name}.png",
            temperature_range,
            pressure_range,
            title,
            7.0,
        )

    rows = compare(composition, reference_points, settings)
    report = dict(source=SOURCE, figure=f"S3-{figure}", interior_assemblages=rows)
    if not check_only:
        report.update(
            liquidus=compare_liquidus(data, *liquidus),
            resolved=data["resolved"],
            diagnostics=data["diagnostics"],
        )
        geometry = bm.phase_diagram_field_polygons(data)
        report["geometry"] = dict(
            fields=len(geometry.polygons),
            open_fields=sum(p.has_open_boundary for p in geometry.polygons),
            unidentified_fields=sum(not p.phases for p in geometry.polygons),
            covered_area_fraction=sum(p.area for p in geometry.polygons),
            diagnostics=list(geometry.diagnostics),
        )
        print(f"Liquidus comparison: {report['liquidus']}", flush=True)
    (output_dir / f"{name}_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"PDF interior assemblages: {sum(row['match'] for row in rows)}/{len(rows)} match.",
        flush=True,
    )
    if not all(row["match"] for row in rows):
        raise RuntimeError("Assemblages differ from the PDF; see the comparison JSON.")
