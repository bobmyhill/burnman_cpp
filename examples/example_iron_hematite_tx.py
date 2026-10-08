#!/usr/bin/env python3
"""Iron-hematite T-X diagram at 1 bar, from Fe to Fe2O3, 300 to 1500 K.

Uses HGP18 iron, wuestite, magnetite and hematite with the Holland
and Powell EOS. X mixes one mole Fe with one mole Fe2O3: Fe=1+X, O=3X.
The system is closed to oxygen. All equilibrium and tracing run in C++.
These are stoichiometric solids; iron allotropes, nonstoichiometric FeO and
liquids are outside this example's candidate set.

Run: python examples/example_iron_hematite_tx.py
Use --quick for fewer seeds, or --no-plots to save only the resumable JSON.
"""

import argparse
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import HGP18

IRON = bm.Composition({"Fe": 1.0}, "molar")
HEMATITE = bm.Composition({"Fe2O3": 1.0}, "molar")


def candidate_phases():
    phases = [
        HGP18.iron(),
        HGP18.wu(),
        HGP18.mt(),
        HGP18.hem(),
    ]
    for phase, name in zip(phases, ["Fe", "FeO", "Fe3O4", "Fe2O3"]):
        phase.set_name(name)
    return phases


def calculate(quick=False, verbose=False):
    domain = dict(temperature_range=(300.0, 1500.0), pressure=1.0e5)
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 5 if quick else 9
    settings.composition_seeds = 5 if quick else 17
    settings.max_phase_instances = 1
    settings.verbose = verbose
    return bm.pseudosection(
        IRON.atomic_composition,
        candidate_phases(),
        diagram="TX",
        composition_end=HEMATITE.atomic_composition,
        settings=settings,
        **domain,
    )


def main():
    stem = "Fe-O_tx"
    condition = "1 bar"
    parser = argparse.ArgumentParser(
        description=f"Iron-Hematite T-X diagram at {condition}."
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
    result = calculate(args.quick, args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bm.save_pseudosection(result, args.output_dir / f"{stem}.json")
    if not args.no_plots:
        fig, ax = bm.plot_pseudosection(
            result,
            temperature_unit="K",
            composition_label="Molar hematite fraction in Fe–Fe$_2$O$_3$ bulk",
            label_fontsize=args.label_fontsize,
            label_key_path=args.output_dir / f"{stem}_labels.md",
            show_nodes=False,
        )
        ax.set_title(f"Iron-Hematite at {condition}")
        fig.savefig(args.output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    print(f"{len(result.boundaries)} phase boundaries; resolved={result.resolved}")
    for diagnostic in result.diagnostics:
        print(diagnostic)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
