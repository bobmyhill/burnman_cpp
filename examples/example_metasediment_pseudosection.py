#!/usr/bin/env python3
"""Closed-system shaly metasediment, 300–900 °C and 0–20 kbar.

All thermodynamics, Gibbs minimisation, phase selection, equilibrium solves
and boundary continuation run in C++. Python specifies the model/bulk and
writes JSON and figures. No Python BurnMan or optimisation library is needed.

The alumina-rich shale contains exactly 2 wt% H2O in a 100 g wet
bulk. FeO and Fe2O3 specify its closed oxygen inventory; water and oxygen are
not buffered. The matched HPx metapelite set reproduces the 23 January 2022
THERMOCALC calibration: White et al. (2014) solids and granitic melt, HP
dataset 6.2, and PS94 H2O with the Holland-Powell thermal reference. It is
available as burnman_cpp.minerals.model_sets.metapelite().

One ternary feldspar model represents plagioclase and alkali feldspar by
coexisting solution instances; fsp means feldspar and 2fsp means two
compositions. ms denotes the white-mica solution (including Na/Ca-bearing
endmembers); 2ms likewise denotes two compositions. Do not add k4tr as a
second candidate: it duplicates pl4tr's thermodynamic model.

Calculations start at 1 bar to avoid the fluid singularity at zero pressure;
figures display 0–20 kbar. Increase --seeds to check narrow/disconnected fields.
--refine-json resumes unfinished lines from accepted native equilibrium states.
With --refine-json, --resolution NX NY also subdivides phase lines using verified
equilibria; finer chords can resolve fields narrower than the saved line spacing.
"""

import argparse
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets

MODEL_SET = model_sets.metapelite()

# Masses in a 100 g wet rock; the sum is 100, including exactly 2 g of H2O.
METASEDIMENT_OXIDES = dict(
    SiO2=61.5,
    Al2O3=20.0,
    FeO=5.0,
    Fe2O3=1.0,
    MgO=2.5,
    CaO=1.0,
    Na2O=2.0,
    K2O=4.0,
    TiO2=1.0,
    H2O=2.0,
)
METASEDIMENT_COMPOSITION = bm.Composition(
    {oxide: grams * 1.0e-3 for oxide, grams in METASEDIMENT_OXIDES.items()},
    unit_type="mass",
)


def candidate_phases():
    return model_sets.metapelite().phases


def calculate(seeds=7, quick=False, verbose=False):
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 4 if quick else seeds
    settings.step = 0.04 if quick else 0.025
    settings.verbose = verbose
    return bm.pseudosection(
        METASEDIMENT_COMPOSITION.atomic_composition,
        candidate_phases(),
        (1.0e5, 2.0e9),
        (573.15, 1173.15),
        settings,
    )


def refine(source, verbose=False, resolution=None):
    """Resume saved phase lines and optionally refine their point spacing."""
    data = bm.load_pseudosection(source)
    settings = bm.PseudosectionResult.from_dict(data).settings
    settings.verbose = verbose
    return bm.refine_pseudosection(
        METASEDIMENT_COMPOSITION.atomic_composition,
        candidate_phases(),
        data,
        settings,
        resolution=resolution,
    )


def save_json(result, path):
    bm.save_pseudosection(
        result,
        path,
        metadata=dict(
            oxide_wt_percent=METASEDIMENT_OXIDES,
            closed_system=True,
            model_set=MODEL_SET.to_dict(),
        ),
    )


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
        title="Shaly metasediment + 2 wt% H₂O, closed system",
    )
    ax.grid(alpha=0.12)
    scope = "HPx metapelite 2022-01-23 + ds62 + PS94 H₂O"
    diagnostics = (
        result["diagnostics"] if isinstance(result, dict) else result.diagnostics
    )
    polygons = ax.pseudosection_geometry.polygons
    open_fields = sum(p.has_open_boundary for p in polygons)
    unidentified = sum(p.n_phases <= 0 or not p.phases for p in polygons)
    status = f"{open_fields} open fields; {unidentified} unidentified fields; {len(diagnostics)} numerical diagnostics in JSON"
    caption = scope + "\n" + status + "; finite seeds may miss narrow fields."
    caption += (
        "\nms: white mica; fsp: feldspar; 2fsp means two compositions; numbered fields: "
        + key_path.name
    )
    fig.text(0.1, 0.015, caption, fontsize=8)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_saved_json(source, output_dir, label_fontsize=7.0):
    data = bm.load_pseudosection(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "metasediment_pseudosection.json"
    if source.resolve() != target.resolve():
        bm.save_pseudosection(data, target)
    plot(data, output_dir / "metasediment_pseudosection.png", label_fontsize)
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
        help="Replot a saved metasediment result without repeating equilibrium solves.",
    )
    parser.add_argument(
        "--refine-json",
        type=Path,
        help="Resume unfinished lines from saved successful equilibrium states in C++.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("metasediment_pseudosection_output")
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        help=(
            "With --refine-json, refine phase lines to these "
            "temperature/pressure axis point counts."
        ),
    )
    args = parser.parse_args()
    if args.resolution is not None:
        if args.refine_json is None:
            parser.error("--resolution requires --refine-json.")
        if any(count < 2 for count in args.resolution):
            parser.error("--resolution counts must be at least 2.")
    if args.plot_json:
        plot_saved_json(args.plot_json, args.output_dir, args.label_fontsize)
        return
    result = (
        refine(args.refine_json, args.verbose, args.resolution)
        if args.refine_json
        else calculate(args.seeds, args.quick, args.verbose)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, args.output_dir / "metasediment_pseudosection.json")
    if not args.no_plots:
        plot(
            result,
            args.output_dir / "metasediment_pseudosection.png",
            args.label_fontsize,
        )
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
