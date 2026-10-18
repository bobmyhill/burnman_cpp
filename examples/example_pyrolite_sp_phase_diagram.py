#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Pyrolite S–P phase diagram, 0–40 GPa, with SLB2024 solids.

Set the entropy bounds from equilibria at (0 GPa, 0 deg C) and
(140 GPa, 4000 deg C), on the original example's 100 g bulk scale.
The upper reference temperature is specified at 140 GPa, not as a
zero-pressure potential temperature. No entropy extrapolation is used.

Calculate directly in P–S coordinates and plot entropy horizontally and
pressure vertically. Start with 41 pressure and entropy seeds, and retain
41 temperature seeds in the settings. The temperature range supplies physical
starting guesses for the entropy-constrained solves; it is not a diagram axis.
The original pyrolite composition, models and EOS-domain policy are retained.
Only solid phases are included; unavailable states remain in the diagnostics.

Run: python examples/example_pyrolite_sp_phase_diagram.py --verbose
Use --quick for four seeds per coordinate, or --plot-json to replot a saved
diagram without calculating equilibria. JSON retains total entropy in J/K;
the plot displays entropy in kB per atom by default.
"""

from math import isfinite
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    load_example,
    plot_phase_diagram_example,
    run_phase_diagram_example,
)

pyrolite = load_example(Path(__file__).with_name("example_pyrolite_phase_diagram.py"))
PRESSURE_RANGE = (0.0, 40.0e9)
ENTROPY_REFERENCES = ((0.0, 273.15), (140.0e9, 4273.15))
TEMPERATURE_RANGE = (273.15, 4273.15)


def candidate_phases():
    """Restore the original pyrolite candidate models in their saved order."""
    return pyrolite.candidate_phases()


def settings(seeds=41, quick=False, verbose=False):
    opts = pyrolite.settings(seeds, quick, verbose)
    opts.entropy_seeds = 41
    opts.pressure_seeds = 41
    return opts


def entropy_references(opts):
    """Return the two reference states' total entropies and their coordinates."""
    records = []
    for pressure, temperature in ENTROPY_REFERENCES:
        state = bm.stable_equilibrium(
            pyrolite.PYROLITE_COMPOSITION.atomic_composition,
            candidate_phases(),
            pressure,
            temperature,
            opts,
        )
        if not state.success or not isfinite(state.entropy):
            raise RuntimeError(
                f"Entropy reference at {pressure / 1e9:g} GPa, "
                f"{temperature - 273.15:g} deg C failed: {state.message}"
            )
        records.append(
            dict(
                pressure_Pa=pressure,
                temperature_K=temperature,
                total_entropy_J_per_K=state.entropy,
                excluded_phases=list(state.excluded_phases),
            )
        )
        print(
            f"Entropy reference at {pressure / 1e9:g} GPa, "
            f"{temperature - 273.15:g} deg C: "
            f"{state.entropy:.8g} J/K per 100 g.",
            flush=True,
        )
    if records[0]["total_entropy_J_per_K"] >= records[1]["total_entropy_J_per_K"]:
        raise ValueError("Reference states must give increasing entropy bounds.")
    return records


def calculate(seeds=41, quick=False, verbose=False):
    """Calculate the native P–S section and retain its entropy-reference metadata."""
    opts = settings(seeds, quick, verbose)
    references = entropy_references(opts)
    entropy_range = tuple(r["total_entropy_J_per_K"] for r in references)
    print(
        f"Entropy bounds: {entropy_range[0]:.8g}–{entropy_range[1]:.8g} J/K per 100 g.",
        flush=True,
    )
    result = bm.phase_diagram(
        pyrolite.PYROLITE_COMPOSITION.atomic_composition,
        candidate_phases(),
        diagram="PS",
        pressure_range=PRESSURE_RANGE,
        entropy_range=entropy_range,
        temperature_range=TEMPERATURE_RANGE,
        settings=opts,
    )
    return result.to_dict() | dict(
        entropy_reference_states=references,
        oxide_wt_percent=pyrolite.PYROLITE_OXIDES,
        initial_ferric_fraction=pyrolite.INITIAL_FERRIC_FRACTION,
        dataset="SLB24",
        dataset_source=pyrolite.PAPER_URL,
        closed_system=True,
        solid_only=True,
    )


def plot(result, path, label_fontsize=7.0, *, entropy_unit="kB/atom"):
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        pressure_unit="GPa",
        entropy_unit=entropy_unit,
        show_nodes=False,
        axes=dict(ylim=(0.0, 40.0), title="Pyrolite, S–P phase diagram"),
        scope="SLB2024 solids; closed oxygen; 3% initial Fe³⁺/ΣFe; no liquids",
        notes="Entropy bounds: S(0 GPa, 0 °C) to S(140 GPa, 4000 °C). Numbered fields: ",
    )


def main():
    parser = example_parser(
        __doc__, "pyrolite_sp_phase_diagram_output", seeds=41, replot=True
    )
    parser.add_argument(
        "--entropy-unit", choices=["kB/atom", "J/K", "kJ/K"], default="kB/atom"
    )
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2.")
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.seeds, args.quick, args.verbose),
        stem="pyrolite_sp_phase_diagram",
        plot=lambda result, path, fontsize: plot(
            result, path, fontsize, entropy_unit=args.entropy_unit
        ),
        args=args,
    )


if __name__ == "__main__":
    main()
