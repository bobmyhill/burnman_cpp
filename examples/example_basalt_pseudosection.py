#!/usr/bin/env python3
"""Closed-system basalt pseudosection, 300–900 °C and 0–20 kbar.

All thermodynamics, Gibbs minimisation, phase selection, equilibrium solves
and boundary continuation run in C++. Python specifies the model/bulk and
writes JSON and a figure. No Python BurnMan or optimisation library is needed.

This illustrative model combines mb50 metabasite solids (HP dataset 6.2),
the HGP 2018 hydrous silicate melt (Cr-free), and PS94 pure H2O fluid with a
NIST gas thermal reference. It is not a reproduction of a consistently
calibrated THERMOCALC model set. Replace candidates for quantitative studies.
The closed bulk contains exactly 2 wt% H2O; no water/oxygen buffers are used.
A 1 bar lower calculation limit avoids the fluid singularity at P=0; the
figure uses the requested 0–20 kbar axis. Increase --seeds to check discovery
of narrow or disconnected fields. Failed calculations remain explicit.
"""

import argparse
import json
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import HP_2011_ds62 as HP
from burnman_cpp.minerals import mb50NCKFMASHTO as MB
from burnman_cpp.minerals import HGP_2018_ds633 as HGP

# Masses in a 100 g wet rock; the sum is 100, including exactly 2 g of H2O.
BASALT_OXIDES = dict(
    SiO2=50.0,
    Al2O3=15.5,
    FeO=8.5,
    Fe2O3=1.5,
    MgO=7.5,
    CaO=10.5,
    Na2O=3.0,
    K2O=0.5,
    TiO2=1.0,
    H2O=2.0,
)
BASALT_COMPOSITION = bm.Composition(
    {oxide: grams * 1.0e-3 for oxide, grams in BASALT_OXIDES.items()}, unit_type="mass"
)


def candidate_phases():
    solutions = [
        (MB.hb, "hb"),
        (MB.dio, "cpx"),
        (MB.opx, "opx"),
        (MB.g, "g"),
        (MB.ol, "ol"),
        (MB.pl4tr, "pl"),
        (MB.ep, "ep"),
        (MB.chl, "chl"),
        (MB.bi, "bi"),
        (MB.ilm, "ilm"),
        (HGP.silicate_melt, "melt"),
    ]
    pure = [
        (HP.q, "q"),
        (HP.law, "law"),
        (HP.ru, "ru"),
        (HP.sph, "sph"),
        (bm.water_fluid, "H2O"),
    ]
    phases = []
    for factory, name in solutions + pure:
        phase = factory()
        phase.set_name(name)
        phases.append(phase)
    return phases


def calculate(seeds=7, quick=False, verbose=False):
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 4 if quick else seeds
    settings.step = 0.04 if quick else 0.025
    settings.verbose = verbose
    return bm.pseudosection(
        BASALT_COMPOSITION.atomic_composition,
        candidate_phases(),
        (1.0e5, 2.0e9),
        (573.15, 1173.15),
        settings,
    )


def refine(source, verbose=False):
    """Continue saved unfinished lines using their last successful states."""
    data = json.loads(source.read_text())
    settings = bm.PseudosectionResult.from_dict(data).settings
    settings.verbose = verbose
    return bm.refine_pseudosection(
        BASALT_COMPOSITION.atomic_composition, candidate_phases(), data, settings
    )


def save_json(result, path):
    data = result.to_dict()
    data.update(oxide_wt_percent=BASALT_OXIDES, closed_system=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def plot(result, path, label_fontsize=7.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(bottom=0.20)
    key_path = path.with_name(path.stem + "_assemblages.md")
    bm.plot_pseudosection(
        result,
        ax=ax,
        phase_aliases={"g": "gt"},
        label_fontsize=label_fontsize,
        label_key_path=key_path,
    )
    if ax.get_legend() is not None:
        ax.get_legend().set_bbox_to_anchor((0.0, -0.11))
    ax.set(
        xlim=(300, 900),
        ylim=(0, 20),
        xlabel="Temperature (°C)",
        ylabel="Pressure (kbar)",
        title="Basalt + 2 wt% H₂O, closed system",
    )
    ax.grid(alpha=0.12)
    scope = "mb50 solids + HGP2018 melt + PS94 H₂O; illustrative model set"
    diagnostics = (
        result["diagnostics"] if isinstance(result, dict) else result.diagnostics
    )
    polygons = ax.pseudosection_geometry.polygons
    open_fields = sum(p.has_open_boundary for p in polygons)
    unidentified = sum(p.n_phases <= 0 or not p.phases for p in polygons)
    status = f"{open_fields} open fields; {unidentified} unidentified fields; {len(diagnostics)} numerical diagnostics in JSON"
    caption = scope + "\n" + status + "; finite seeds may miss narrow fields."
    caption += (
        "\n2hb means two hornblende compositions; numbered fields: " + key_path.name
    )
    fig.text(0.1, 0.015, caption, fontsize=8)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_saved_json(source, output_dir, label_fontsize=7.0):
    data = json.loads(source.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "basalt_pseudosection.json"
    if source.resolve() != target.resolve():
        target.write_text(source.read_text())
    plot(data, output_dir / "basalt_pseudosection.png", label_fontsize)
    print(f"Coloured pseudosection figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=7.0,
        help="Font size in points for every assemblage and field number (default: 7).",
    )
    parser.add_argument(
        "--plot-json",
        type=Path,
        help="Replot a saved basalt result without repeating equilibrium solves.",
    )
    parser.add_argument(
        "--refine-json",
        type=Path,
        help="Resume unfinished lines from saved successful equilibrium states in C++.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pseudosection_output"))
    args = parser.parse_args()
    if args.plot_json:
        plot_saved_json(args.plot_json, args.output_dir, args.label_fontsize)
        return
    result = (
        refine(args.refine_json, args.verbose)
        if args.refine_json
        else calculate(args.seeds, args.quick, args.verbose)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, args.output_dir / "basalt_pseudosection.json")
    if not args.no_plots:
        plot(result, args.output_dir / "basalt_pseudosection.png", args.label_fontsize)
    print(
        f"{len(result.fields)} fields, {len(result.boundaries)} edges, "
        f'{sum(n.kind=="junction" for n in result.nodes)} junctions; '
        f"{result.equilibrium_solves} native equilibrium solves."
    )
    print(f"Resolved: {result.resolved}. Outputs: {args.output_dir}")
    for diagnostic in result.diagnostics:
        print(diagnostic)


if __name__ == "__main__":
    main()
