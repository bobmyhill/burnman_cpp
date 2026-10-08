"""Shared runner for Supplement 3 of Green et al. (2025), egae079.

Use the separate example_green2025_*.py scripts. Thermodynamics, equilibrium,
boundary tracing and polygon construction use the standard C++ routines.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import burnman_cpp as bm
from burnman_cpp.minerals import HPx_ds636, model_sets

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
    return dict(
        pressure_kbar=pressure_kbar,
        reference_temperature_C=reference_temperature,
        reference_reading_tolerance_C=5.0,
        calculated_temperature_C=calculated,
        difference_C=None if calculated is None else calculated - reference_temperature,
        match=calculated is not None and abs(calculated - reference_temperature) <= 5.0,
    )


def plot(data, target, temperature_range, pressure_range, title, fontsize):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    bm.plot_pseudosection(
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
    fig.savefig(target, dpi=180)
    fig.savefig(target.with_suffix(".svg"))
    plt.close(fig)


def main(
    name,
    title,
    figure,
    oxides,
    temperature_range,
    pressure_range,
    reference_points,
    liquidus,
):
    parser = argparse.ArgumentParser(
        description=f"Green et al. (2025), Figure S3-{figure}: {title}"
    )
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument(
        "--quick", action="store_true", help="Coarse discovery; may miss narrow fields."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=7.0)
    parser.add_argument("--plot-json", type=Path)
    parser.add_argument("--refine-json", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only compare interior assemblages with the PDF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / f"green2025_{name}_output",
    )
    args = parser.parse_args()
    # These are oxide moles (including atomic O), not oxide weight percentages.
    # Normalising the amount improves numerical scaling without changing the bulk.
    composition = bm.Composition(
        {oxide: amount / sum(oxides.values()) for oxide, amount in oxides.items()},
        unit_type="molar",
    )
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = (
        3 if args.quick else args.seeds
    )
    settings.step = 0.035 if args.quick else 0.015
    settings.max_recovery_passes = 0 if args.quick else 2
    settings.verbose = args.verbose
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.plot_json:
        data = bm.load_pseudosection(args.plot_json)
    elif not args.check_only:
        phases = candidate_phases(composition)
        result = (
            bm.refine_pseudosection(
                composition.atomic_composition,
                phases,
                bm.load_pseudosection(args.refine_json),
                settings,
            )
            if args.refine_json
            else bm.pseudosection(
                composition.atomic_composition,
                phases,
                (max(1.0e5, pressure_range[0] * 1.0e8), pressure_range[1] * 1.0e8),
                tuple(t + 273.15 for t in temperature_range),
                settings,
            )
        )
        data = result.to_dict()
        data.update(
            source=SOURCE,
            reference_figure=f"S3-{figure}",
            oxide_molar_amounts=oxides,
            model_set=model_sets.igneous("G25").to_dict(),
            calculation_seconds=time.monotonic() - started,
        )
        bm.save_pseudosection(data, args.output_dir / f"{name}.json")
        print(
            f"{len(result.boundaries)} boundaries; resolved={result.resolved}; "
            f"{result.equilibrium_solves} solves in {data['calculation_seconds']:.1f} s",
            flush=True,
        )
    if not args.check_only and not args.no_plots:
        plot(
            data,
            args.output_dir / f"{name}.png",
            temperature_range,
            pressure_range,
            title,
            args.label_fontsize,
        )
    if not args.plot_json:
        rows = compare(composition, reference_points, settings)
        report = dict(source=SOURCE, figure=f"S3-{figure}", interior_assemblages=rows)
        if not args.check_only:
            report["liquidus"] = compare_liquidus(data, *liquidus)
            report["resolved"] = data["resolved"]
            report["diagnostics"] = data["diagnostics"]
            geometry = bm.pseudosection_field_polygons(data)
            report["geometry"] = dict(
                fields=len(geometry.polygons),
                open_fields=sum(p.has_open_boundary for p in geometry.polygons),
                unidentified_fields=sum(not p.phases for p in geometry.polygons),
                covered_area_fraction=sum(p.area for p in geometry.polygons),
                diagnostics=list(geometry.diagnostics),
            )
            print(f"Liquidus comparison: {report['liquidus']}", flush=True)
        (args.output_dir / f"{name}_comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        print(
            f"PDF interior assemblages: {sum(row['match'] for row in rows)}/{len(rows)} match.",
            flush=True,
        )
        if not all(row["match"] for row in rows):
            raise RuntimeError(
                "Assemblages differ from the PDF; see the comparison JSON."
            )
