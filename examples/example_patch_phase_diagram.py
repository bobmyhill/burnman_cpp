#!/usr/bin/env python3
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

"""Calculate local patches and optionally insert them into a saved diagram.

Four modes (Field numbers are the one-based numbers shown on plots):
    make-for-fields SOURCE --fields 27 28 --models MODEL --seeds 9 11
    make-for-problems SOURCE --models MODEL --seeds 9 11
    make-and-patch SOURCE --models MODEL [--fields 27 28] --seeds 9 11
    patch SOURCE PATCH_JSON [PATCH_JSON ...]

The first two modes only save patches. make-for-problems includes provisional,
unlabelled and open fields, plus boundaries flagged in saved diagnostics.
make-and-patch uses the same selection unless --fields is supplied. Missing
sampled assemblages do not define patch windows automatically.

--padding expands each window (default 5 percent per side). --seeds
sets the local x/y seed counts: temperature/pressure for PT diagrams.
Other solver settings are reused from SOURCE. MODEL is a module name or
Python file supplying candidate_phases(). Each patch is saved under
field_NUMBER_patch or boundary_ID_patch; boundary IDs match the diagnostics.
Merging also saves final/phase_diagram.json, figures and a merge report.
Existing patches are applied in the order supplied, so the last patch wins
in overlaps.
"""

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import burnman_cpp as bm


def candidate_phases(module_name):
    """Accept an import name or a Python file, including sibling imports."""
    path = Path(module_name)
    if path.suffix != ".py":
        return importlib.import_module(module_name).candidate_phases()
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
        return module.candidate_phases()
    finally:
        sys.path.pop(0)


def save(result, directory, plots, metadata=None):
    data = result if isinstance(result, dict) else result.to_dict()
    data = dict(metadata or {}) | data
    bm.save_phase_diagram(data, directory / "phase_diagram.json")
    if plots:
        import matplotlib.pyplot as plt

        figure, _ = bm.plot_phase_diagram(
            data, label_key_path=directory / "assemblages.md"
        )
        figure.savefig(directory / "phase_diagram.png", dpi=200)
        figure.savefig(directory / "phase_diagram.svg")
        plt.close(figure)
    print(f"Resolved: {data['resolved']}. Outputs: {directory}", flush=True)
    for message in data.get("diagnostics", []):
        print(message, flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("make-for-fields", "Save patches for named fields."),
        ("make-for-problems", "Save patches for problematic fields and boundaries."),
        ("make-and-patch", "Calculate patches, then insert them into the diagram."),
        ("patch", "Insert existing patch JSON files."),
    ):
        command = modes.add_parser(mode, help=help_text, description=help_text)
        command.add_argument("source", type=Path)
        command.add_argument(
            "--models",
            required=mode != "patch",
            help="Module or .py file supplying candidate_phases(); optional native checks in patch mode.",
        )
        command.add_argument(
            "--output-dir", type=Path, default=Path("patched_phase_diagram_output")
        )
        command.add_argument("--no-plots", action="store_true")
        if mode == "patch":
            command.add_argument(
                "patches",
                nargs="+",
                type=Path,
                help="Patch JSON files in insertion order.",
            )
        else:
            command.add_argument(
                "--seeds",
                nargs=2,
                type=int,
                default=(9, 9),
                metavar=("NX", "NY"),
                help="Local x/y seed counts (default 9 9); PT uses temperature/pressure.",
            )
            command.add_argument(
                "--padding",
                type=float,
                default=5.0,
                help="Window padding per side in percent (default 5).",
            )
            command.add_argument("--verbose", action="store_true")
            if mode != "make-for-problems":
                command.add_argument(
                    "--fields",
                    nargs="+",
                    type=int,
                    required=mode == "make-for-fields",
                    help="One-based Field numbers; make-and-patch defaults to problematic fields and boundaries.",
                )
    args = parser.parse_args()
    source = bm.load_phase_diagram(args.source)
    if args.mode == "patch":
        phases = candidate_phases(args.models) if args.models else None
        patches = [(str(path), bm.load_phase_diagram(path)) for path in args.patches]
    else:
        if min(args.seeds) < 2:
            parser.error("--seeds counts must each be at least 2.")
        fields = getattr(args, "fields", None)
        windows = (
            {
                f"Field {n}": region
                for n, region in bm.phase_diagram_field_regions(
                    source, fields, padding=args.padding / 100
                ).items()
            }
            if fields is not None
            else bm.phase_diagram_problem_regions(source, padding=args.padding / 100)
        )
        if not windows:
            print(
                "No problematic fields or boundaries found; no patches were calculated."
            )
            return
        phases = candidate_phases(args.models)
        settings = bm.PhaseDiagramSettings.from_dict(source.get("settings", {}))
        names = dict(
            P="pressure", T="temperature", S="entropy", V="volume", X="composition"
        )
        for axis, count in zip(source.get("diagram_type", "PT")[::-1], args.seeds):
            setattr(settings, names[axis] + "_seeds", count)
        settings.verbose = args.verbose
        patches = []
        for label, region in windows.items():
            print(f"Calculating {label}: x={region[0]}, y={region[1]}", flush=True)
            patch = bm.calculate_phase_diagram_patch(source, phases, region, settings)
            patch = save(
                patch,
                args.output_dir / (label.lower().replace(" ", "_") + "_patch"),
                not args.no_plots,
                metadata=source,
            )
            patches.append((label, patch))
        if args.mode != "make-and-patch":
            return
    merged, reports = source, []
    for label, patch in patches:
        merged, report = bm.merge_phase_diagrams(merged, patch, phases=phases)
        for message in report:
            message = f"{label}: {message}"
            reports.append(message)
            print(message, flush=True)
    save(merged, args.output_dir / "final", not args.no_plots)
    (args.output_dir / "merge_report.txt").write_text(
        "\n".join(reports) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
