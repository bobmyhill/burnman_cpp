#!/usr/bin/env python3
"""Mg2SiO4 phase fields in P-T, P-S, V-T and V-S coordinates.

Recreate the phase boundaries in Bob Myhill's thermodynamics illustration:
https://blogs.egu.eu/divisions/gd/2021/01/27/thermodynamics-and-geodynamics-the-perfect-couple/

Use the SLB2011 periclase, bridgmanite, akimotoite and ringwoodite models.
Each panel is calculated independently by the native pseudosection routine;
the shared plotter displays volume as density and entropy in kB per atom.
TV and SV are the canonical names of the V-T and V-S section types.

Run: python examples/example_forsterite_thermodynamic_sections.py
Use --volume-unit m3 --entropy-unit J/K to display total volume and entropy.
"""

import argparse
import json
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB11

BULK = bm.Composition({"Mg2SiO4": 1.0}, "molar")
BULK_MASS = sum(BULK.mass_composition.values())
ENTROPY_SCALE = sum(BULK.atomic_composition.values()) * 8.31446261815324
ALIASES = ["per", "bdg", "aki", "rw"]


def candidate_phases():
    phases = [
        SLB11.periclase(),
        SLB11.mg_perovskite(),
        SLB11.mg_akimotoite(),
        SLB11.mg_ringwoodite(),
    ]
    for phase, name in zip(phases, ALIASES):
        phase.set_name(name)
    return phases


def calculate(quick=False, verbose=False):
    pressure_range = (23.2e9, 23.9e9)
    temperature_range = (1400.0, 1700.0)
    entropy_range = tuple(s * ENTROPY_SCALE for s in (5.2, 5.8))
    volume_range = tuple(BULK_MASS / rho for rho in (4500.0, 3600.0))
    domains = {
        "PT": dict(pressure_range=pressure_range, temperature_range=temperature_range),
        "PS": dict(
            pressure_range=pressure_range,
            entropy_range=entropy_range,
            temperature_range=(500.0, 3000.0),
        ),
        "TV": dict(
            temperature_range=temperature_range,
            volume_range=volume_range,
            pressure_range=(0.0, 100.0e9),
        ),
        "SV": dict(
            entropy_range=entropy_range,
            volume_range=volume_range,
            pressure_range=(0.0, 100.0e9),
            temperature_range=(500.0, 3000.0),
        ),
    }
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = settings.entropy_seeds = (
        settings.volume_seeds
    ) = (5 if quick else 9)
    settings.max_phase_instances = 1
    settings.verbose = verbose
    results = {}
    for diagram, domain in domains.items():
        results[diagram] = bm.pseudosection(
            BULK.atomic_composition,
            candidate_phases(),
            diagram=diagram,
            settings=settings,
            **domain,
        )
    return results


def plot(
    results,
    output_dir,
    label_fontsize=9.0,
    *,
    volume_unit="kg/m3",
    entropy_unit="kB/atom",
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colours = {
        frozenset([0, 1, 2]): "#bdd7ee",
        frozenset([0, 2, 3]): "#cde6d5",
        frozenset([0, 1, 3]): "#edb4a9",
        frozenset([0, 1, 2, 3]): "#cbb6e6",
    }
    for ax, (diagram, result) in zip(axes.flat, results.items()):
        bm.plot_pseudosection(
            result,
            ax=ax,
            colorbar=False,
            fill_alpha=1.0,
            pressure_unit="GPa",
            temperature_unit="K",
            entropy_unit=entropy_unit,
            volume_unit=volume_unit,
            swap_axes=diagram in ("TV", "SV"),
            label_fontsize=label_fontsize,
            label_key_path=output_dir / f"forsterite_{diagram.lower()}_labels.md",
        )
        polygons = [
            p
            for p in ax.pseudosection_geometry.polygons
            if p.n_phases > 0 and not p.outside_model_domain
        ]
        # Colour the coexistence fields by assemblage, as in the illustration.
        for collection in ax.collections:
            if isinstance(collection, PatchCollection):
                collection.set_array(None)
                collection.set_facecolors(
                    [colours.get(frozenset(p.phases), "#e5e5e5") for p in polygons]
                )
                break
        ax.set_axisbelow(True)
        ax.grid(color="white", linewidth=1.0)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=9.0)
    parser.add_argument(
        "--volume-unit", choices=["kg/m3", "m3", "cm3"], default="kg/m3"
    )
    parser.add_argument(
        "--entropy-unit", choices=["kB/atom", "J/K", "kJ/K"], default="kB/atom"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "forsterite_sections_output",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = calculate(args.quick, args.verbose)
    for diagram, result in results.items():
        (args.output_dir / f"forsterite_{diagram.lower()}.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        print(
            f"{diagram}: {len(result.boundaries)} boundaries; resolved={result.resolved}"
        )
        for diagnostic in result.diagnostics:
            print(diagnostic)
    if not args.no_plots:
        fig, _ = plot(
            results,
            args.output_dir,
            args.label_fontsize,
            volume_unit=args.volume_unit,
            entropy_unit=args.entropy_unit,
        )
        fig.savefig(args.output_dir / "forsterite_sections.png", dpi=200)
        fig.savefig(args.output_dir / "forsterite_sections.pdf")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
