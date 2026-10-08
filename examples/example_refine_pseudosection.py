#!/usr/bin/env python3
"""Refine all phase lines in a saved diagram, then save JSON and a figure.

Use the original example's candidate_phases() to restore its models, e.g.:
python examples/example_refine_pseudosection.py metasediment_pseudosection.json \
    --models example_metasediment_pseudosection --resolution 101 201

Resolution counts axis points: here 100 divisions of x and 200 divisions of y.
All new equilibrium points are calculated and validated in C++.
Completed diagrams reuse saved lines; recovery runs where checks find problems.
Add --verbose to report line refinement and field-checking progress.
"""

import argparse
import importlib
from pathlib import Path

import burnman_cpp as bm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--models",
        required=True,
        help="Original example module supplying candidate_phases().",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        required=True,
        metavar=("NX", "NY"),
        help="Axis point counts (at least 2); 101 201 gives 100 x and 200 y divisions.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("refined_pseudosection_output")
    )
    for name, default, choices in (
        ("pressure", "kbar", ("Pa", "GPa", "kbar")),
        ("temperature", "C", ("C", "K")),
        ("entropy", "J/K", ("J/K", "kJ/K", "kB/atom")),
        ("volume", "m3", ("m3", "cm3", "kg/m3")),
    ):
        parser.add_argument(f"--{name}-unit", default=default, choices=choices)
    parser.add_argument("--swap-axes", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=7.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Print refinement progress."
    )
    args = parser.parse_args()
    saved = bm.load_pseudosection(args.source)
    candidates = importlib.import_module(args.models).candidate_phases()
    settings = bm.PseudosectionSettings.from_dict(saved.get("settings", {}))
    settings.verbose = args.verbose
    if args.verbose:
        print(f"Loaded {args.source}. Starting refinement.", flush=True)
    display = {
        key: getattr(args, key)
        for key in (
            "pressure_unit",
            "temperature_unit",
            "entropy_unit",
            "volume_unit",
            "swap_axes",
        )
    }
    result = bm.refine_pseudosection(
        saved["composition_start"],
        candidates,
        saved,
        settings,
        resolution=args.resolution,
        **display,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Saving refined diagram to {args.output_dir}.", flush=True)
    bm.save_pseudosection(
        result, args.output_dir / "pseudosection.json", metadata=saved
    )
    if not args.no_plots:
        import matplotlib.pyplot as plt

        if args.verbose:
            print("Plotting refined diagram.", flush=True)
        figure, _ = bm.plot_pseudosection(
            result,
            **display,
            label_fontsize=args.label_fontsize,
            label_key_path=args.output_dir / "assemblages.md",
        )
        figure.savefig(args.output_dir / "pseudosection.png", dpi=200)
        figure.savefig(args.output_dir / "pseudosection.svg")
        plt.close(figure)
    print(f"Resolved: {result.resolved}. Outputs: {args.output_dir}")
    for diagnostic in result.diagnostics:
        print(diagnostic)


if __name__ == "__main__":
    main()
