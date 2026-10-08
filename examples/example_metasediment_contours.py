#!/usr/bin/env python3
"""Density and garnet Mg/(Mg + Fe2+) contours on a saved metasediment diagram.

First run example_metasediment_pseudosection.py, then:

    python examples/example_metasediment_contours.py path/to/metasediment_pseudosection.json

The saved phase lines, closed fields and equilibrium states are reused.
Native VolumeConstraint sets total volume to bulk mass / density.
Native PhaseCompositionConstraint selects garnet's Mg and ferrous-Fe sites;
its ferric-Fe site is excluded. Seeding, correction and continuation use C++.
Contour JSON can subsequently be plotted with --plot-json, without solving.
"""

import argparse
import json
from pathlib import Path

import burnman_cpp as bm
from example_metasediment_pseudosection import (
    METASEDIMENT_COMPOSITION,
    candidate_phases,
)


def calculate(diagram, density_levels, garnet_levels, seed_grid=5, step=0.02):
    phases = candidate_phases()
    settings = bm.PseudosectionContourSettings()
    settings.seed_grid = seed_grid
    settings.step = step
    composition = diagram.get(
        "composition_start", METASEDIMENT_COMPOSITION.atomic_composition
    )
    # The Composition object keeps density on the saved bulk's amount scale.
    mass = sum(bm.Composition(composition, "molar").mass_composition.values())
    # MP14 garnet's A sites contain Mg and Fe2+; its Fe3+ B site is excluded.
    constructors = {
        "density": (density_levels, lambda value: bm.VolumeConstraint(mass / value)),
        "garnet": (
            garnet_levels,
            lambda value: bm.PhaseCompositionConstraint.for_diagram(
                diagram, "g", ["Mgx_A", "Fex_A"], [1.0, 0.0], [1.0, 1.0], value
            ),
        ),
    }
    return {
        name: bm.pseudosection_contour_levels(
            diagram,
            phases,
            levels,
            constructor,
            settings=settings,
            composition=composition,
            verbose=True,
        )
        for name, (levels, constructor) in constructors.items()
    }


def plot(diagram, contours, output_dir, label_fontsize=7.0):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(19, 9), layout="constrained")
    for ax, name in zip(axes, ["density", "garnet"]):
        bm.plot_pseudosection(
            diagram,
            ax=ax,
            phase_aliases={"g": "gt"},
            fill_alpha=0.2,
            label_fontsize=label_fontsize,
            label_key_path=output_dir / f"{name}_assemblages.md",
        )
        for record in contours[name]:
            bm.plot_pseudosection_contours(
                diagram,
                record["contours"],
                ax,
                color="crimson" if name == "density" else "royalblue",
                value=record["value"],
                label_fontsize=label_fontsize,
            )
        ax.set_title(
            "Bulk density (kg/m³)"
            if name == "density"
            else r"Garnet molar Mg / (Mg + Fe$^{2+}$)"
        )
        ax.set_xlim(300.0, 900.0)
        ax.set_ylim(0.0, 20.0)
    fig.savefig(output_dir / "metasediment_contours.png", dpi=180)
    fig.savefig(output_dir / "metasediment_contours.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pseudosection_json", type=Path)
    parser.add_argument(
        "--density", type=float, nargs="+", default=[2600.0, 2800.0, 3000.0, 3200.0]
    )
    parser.add_argument(
        "--garnet", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3]
    )
    parser.add_argument("--seed-grid", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--label-fontsize", type=float, default=7.0)
    parser.add_argument(
        "--plot-json", type=Path, help="Plot previously saved contours without solving."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("metasediment_contours_output")
    )
    args = parser.parse_args()
    if any(v <= 0.0 for v in args.density):
        parser.error("Density levels must be positive.")
    if any(v <= 0.0 or v >= 1.0 for v in args.garnet):
        parser.error("Garnet Mg/(Mg+Fe2+) levels must be between zero and one.")
    diagram = bm.load_pseudosection(args.pseudosection_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contours = (
        json.loads(args.plot_json.read_text())
        if args.plot_json
        else calculate(diagram, args.density, args.garnet, args.seed_grid, args.step)
    )
    (args.output_dir / "metasediment_contours.json").write_text(
        json.dumps(contours, indent=2, allow_nan=False)
    )
    plot(diagram, contours, args.output_dir, args.label_fontsize)


if __name__ == "__main__":
    main()
