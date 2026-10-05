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


def garnet_constraint(value, phase_names):
    """Build the constraint using garnet's index in the current field."""

    def factory(a, prm, ids):
        # Saved phase_names includes solution-instance suffixes, indexed by ID.
        indices = [
            i
            for i, phase_id in enumerate(ids)
            if phase_names[phase_id].split(" #")[0] == "g"
        ]
        if not indices:
            return None
        index = indices[0]
        phase = a.get_phase(index)
        # mp50 garnet: three Mg/Fe2+/Ca sites (A), Al/Fe3+ sites (B).
        # n_occupancies in PhaseCompositionConstraint supplies multiplicities.
        sites = ["Mgx_A", "Fex_A"]
        if not isinstance(phase, bm.Solution) or not all(
            site in phase.site_names for site in sites
        ):
            return None  # An active pure face cannot carry this ratio.
        return bm.PhaseCompositionConstraint(
            index,
            sites,
            [1.0, 0.0],
            [1.0, 1.0],
            value,
            a,
            prm,
        )

    return factory


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
    output = dict(density=[], garnet=[])
    for name, levels in [("density", density_levels), ("garnet", garnet_levels)]:
        for value in levels:
            constraint = (
                bm.VolumeConstraint(mass / value)
                if name == "density"
                else garnet_constraint(value, diagram["phase_names"])
            )
            contours = bm.pseudosection_contours(
                diagram,
                phases,
                constraint,
                settings=settings,
                composition=composition,
            )
            print(
                f"{name} {value:g}: {len(contours.lines)} segments, "
                f"{contours.equilibrium_solves} native solves, "
                f"{len(contours.diagnostics)} diagnostics",
                flush=True,
            )
            output[name].append(dict(value=value, contours=contours.to_dict()))
    return output


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
                label=f'{record["value"]:g}',
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
    diagram = json.loads(args.pseudosection_json.read_text())
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
