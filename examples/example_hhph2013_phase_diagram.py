#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""
Dry HHPH2013 phase diagram for the paper's Figure 1 KLB-1 bulk.

The model set includes the HHPH2013 high-pressure solutions and
Green et al. pyroxenes. Orthopyroxene (opx) and
high-pressure C2/c pyroxene (hpx) are limited to xCa(M2) <= 0.2.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    plot_phase_diagram_example,
    refine_saved,
    write_phase_diagram,
)

OUTPUT_DIR = Path(__file__).resolve().with_name("hhph2013_phase_diagram_output")
PRESSURE_RANGE = (3.0e9, 30.0e9)
TEMPERATURE_RANGE = (1473.15, 2273.15)  # 1200-2000 °C
SEEDS = 7
MAX_PYROXENE_CA = 0.2  # Maximum Ca content in opx and hpx
REFINEMENT_RESOLUTION = (201, 201)

FIGURE_1_OXIDE_MOLES = dict(SiO2=39.4, Al2O3=2.0, CaO=3.3, MgO=49.5, FeO=5.2, Na2O=0.26)
BULK_COMPOSITION = bm.Composition(FIGURE_1_OXIDE_MOLES, unit_type="molar")
BULK_COMPOSITION.renormalize("mass", "total", 0.1)

MODEL_SET = model_sets.hhph2013()
PHASE_GLOSSARY = MODEL_SET.phase_glossary
PHASES = MODEL_SET.phases
for phase in PHASES:
    if phase.name in ("opx", "hpx"):
        coefficients = [MAX_PYROXENE_CA] * phase.n_endmembers
        coefficients[3] -= 1.0  # odi is the Ca-bearing endmember.
        phase.endmember_constraints = [coefficients]


def settings(seeds=SEEDS, verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = seeds
    opts.verbose = verbose
    return opts


def save_json(result, path):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            figure_1_molar_oxide_bulk=FIGURE_1_OXIDE_MOLES,
            model_set=MODEL_SET.to_dict(),
            max_pyroxene_ca=MAX_PYROXENE_CA,
            closed_system=True,
            solid_only=True,
            ferric_iron_fraction=0.0,
            display_pressure_range_GPa=[p / 1.0e9 for p in PRESSURE_RANGE],
            display_temperature_range_C=[t - 273.15 for t in TEMPERATURE_RANGE],
        ),
    )


def plot(result, path, label_fontsize=7.0):
    bulk = [f"{oxide} {amount:g}" for oxide, amount in FIGURE_1_OXIDE_MOLES.items()]
    title = (
        "KLB-1 (Figure 1): HHPH2013\nBulk (molar oxide amounts): "
        + ", ".join(bulk[:3])
        + "\n"
        + ", ".join(bulk[3:])
    )
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        pressure_unit="GPa",
        temperature_unit="C",
        show_nodes=False,
        axes=dict(
            xlim=tuple(t - 273.15 for t in TEMPERATURE_RANGE),
            ylim=tuple(p / 1.0e9 for p in PRESSURE_RANGE),
            xlabel="Temperature (°C)",
            ylabel="Pressure (GPa)",
            title=title,
        ),
        scope="HHPH2013 archive; opx/hpx xCa(M2) ≤ 0.2; all Fe²⁺; no melt or fluid",
        notes="Numbered fields: ",
    )
    key_path = Path(path).with_name(Path(path).stem + "_assemblages.md")
    if key_path.exists():
        with key_path.open("a") as key:
            key.write("\n## Phase abbreviations\n\n| Label | Model |\n| --- | --- |\n")
            for label, name in PHASE_GLOSSARY.items():
                key.write(f"| {label} | {name} |\n")


if __name__ == "__main__":
    phase_diagram = bm.phase_diagram(
        BULK_COMPOSITION.atomic_composition,
        PHASES,
        PRESSURE_RANGE,
        TEMPERATURE_RANGE,
        settings(verbose=True),
    )
    stem = OUTPUT_DIR / "hhph2013_phase_diagram"
    write_phase_diagram(
        phase_diagram,
        stem,
        save_json=save_json,
        plot=plot,
    )
    refined = refine_saved(
        stem.with_suffix(".json"),
        PHASES,
        resolution=REFINEMENT_RESOLUTION,
        verbose=True,
    )
    write_phase_diagram(
        refined,
        OUTPUT_DIR / "hhph2013_phase_diagram_201x201",
        save_json=save_json,
        plot=plot,
    )
