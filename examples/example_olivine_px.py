#!/usr/bin/env python3
"""Olivine-polymorph P-X diagram at 1673 K, from Mg2SiO4 to Fe2SiO4.

The candidate set comprises olivine, wadsleyite and ringwoodite from SLB11.
X is the molar fayalite fraction Fe/(Mg+Fe). All stability searches and phase-
boundary continuation run in C++; plotting uses the standard pseudosection API.

Run: python examples/example_olivine_px.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

import argparse
import json
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB11

FORSTERITE = bm.Composition({"Mg2SiO4": 1.0}, "molar")
FAYALITE = bm.Composition({"Fe2SiO4": 1.0}, "molar")


def candidate_phases():
    phases = [
        SLB11.mg_fe_olivine(),
        SLB11.mg_fe_wadsleyite(),
        SLB11.mg_fe_ringwoodite(),
    ]
    for phase, name in zip(phases, ["ol", "wa", "ri"]):
        phase.set_name(name)
    return phases


def calculate(quick=False, verbose=False, *, diagram="PX"):
    domains = {
        "PX": dict(pressure_range=(0.0, 20.0e9), temperature=1673.0),
        "TX": dict(temperature_range=(1000.0, 3000.0), pressure=14.0e9),
        "VX": dict(volume_range=(36.0e-6, 45.0e-6), temperature=1673.0),
    }
    if diagram not in domains:
        raise ValueError("The olivine example supports PX, TX or VX diagrams.")
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 5 if quick else 9
    settings.volume_seeds = settings.pressure_seeds
    settings.composition_seeds = 5 if quick else 17
    settings.max_phase_instances = 1
    settings.verbose = verbose
    return bm.pseudosection(
        FORSTERITE.atomic_composition,
        candidate_phases(),
        diagram=diagram,
        composition_end=FAYALITE.atomic_composition,
        settings=settings,
        **domains[diagram],
    )


def main(diagram="PX"):
    stem = f"olivine_{diagram.lower()}"
    condition = "14 GPa" if diagram == "TX" else "1673 K"
    parser = argparse.ArgumentParser(
        description=f"Olivine-polymorph {diagram[0]}-X diagram at {condition}."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=9.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / f"{stem}_output",
    )
    args = parser.parse_args()
    result = calculate(args.quick, args.verbose, diagram=diagram)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n"
    )
    if not args.no_plots:
        fig, ax = bm.plot_pseudosection(
            result,
            pressure_unit="GPa",
            temperature_unit="K",
            volume_unit="cm3",
            composition_label=r"$X_{\mathrm{Fe}} = \mathrm{Fe}/(\mathrm{Mg}+\mathrm{Fe})$",
            phase_aliases={"ol": "ol", "wa": "wa", "ri": "ri"},
            label_fontsize=args.label_fontsize,
            label_key_path=args.output_dir / f"{stem}_labels.md",
            show_nodes=True,
        )
        if diagram == "VX":
            ax.set_ylabel("Volume (cm³ per mole of bulk formula units)")
        ax.set_title(f"Olivine polymorphs at {condition}")
        fig.savefig(args.output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    print(f"{len(result.boundaries)} phase boundaries; resolved={result.resolved}")
    for diagnostic in result.diagnostics:
        print(diagnostic)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
