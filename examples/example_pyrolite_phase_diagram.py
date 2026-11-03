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

"""
Phase diagram generation for a dry, solid, closed-system pyrolite containing
2% Fe³⁺/ΣFe on a metal-free basis, calculated from 0-25 GPa and 0-2500 K.
It uses the Stixrude & Lithgow-Bertelloni (2024) thermodynamic dataset,
with a composition taken from Table 1 of that paper.

Two regions of the phase diagram are conservatively masked:
(1) the Mg-Tschermak thermal spinodal makes the full orthopyroxene model
undefined at low pressure and high temperature, and
(2) the calcium-ferrite-structured solution has negative heat capacity
below its 5 K ordering transition.
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    plot_phase_diagram_example,
    write_phase_diagram,
)

OUTPUT_DIR = Path(__file__).resolve().with_name("pyrolite_phase_diagram_output")
SEEDS = 9
PAPER_URL = "https://doi.org/10.1093/gji/ggae126"
MODEL_SET = model_sets.slb2024()
PHASE_GLOSSARY = MODEL_SET.phase_glossary
PHASES = MODEL_SET.phases

PRESSURE_RANGE = (0.0e9, 25.0e9)  # in Pa.
TEMPERATURE_RANGE = (0.0, 2500.0)  # in K.

# Get the pyrolite composition from Table 1 of Stixrude & Lithgow-Bertelloni (2024).
# This corresponds to an Fe3+/sum(Fe) ratio of 2% on a metal-free basis.
# The composition is normalized to 100 g (0.1 kg).
PYROLITE_COMPOSITION = bm.Composition(
        dict(
            SiO2=45.00,
            MgO=38.88,
            FeO=8.05,
            Fe2O3=0.18,
            CaO=3.18,
            Al2O3=4.00,
            Na2O=0.13,
            Cr2O3=0.57,
        ),
        unit_type="mass",
    )
PYROLITE_COMPOSITION.renormalize("mass", "total", 0.1)

# Masses in grams of a 100 g rock are also oxide weight percentages
# for JSON storage.
PYROLITE_OXIDES = {
    oxide: kg * 1.0e3 for oxide, kg in PYROLITE_COMPOSITION.mass_composition.items()
}


def settings(seeds=9, verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = seeds
    # Mask the diagram where the EoS of Mg-Tschermak breaks down.
    opts.required_eos_phases = ["opx"]
    # Mask the diagram where endmembers of cf have negative heat capacities.
    opts.required_thermal_phases = ["cf"]
    opts.verbose = verbose
    return opts


def save_json(
    result, path, *, pressure_range=PRESSURE_RANGE, temperature_range=TEMPERATURE_RANGE
):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            oxide_wt_percent=PYROLITE_OXIDES,
            closed_system=True,
            dataset="SLB_2024",
            dataset_source=PAPER_URL,
            model_set=MODEL_SET.to_dict(),
            solid_only=True,
            display_pressure_range_GPa=[p / 1.0e9 for p in pressure_range],
            display_temperature_range_K=list(temperature_range),
        ),
    )


def plot(
    result,
    path,
    label_fontsize=7.0,
    *,
    pressure_range=PRESSURE_RANGE,
    temperature_range=TEMPERATURE_RANGE,
    figsize=(12, 10),
):
    key_path = Path(path).with_name(Path(path).stem + "_assemblages.md")
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        pressure_unit="GPa",
        temperature_unit="K",
        show_nodes=False,
        figsize=figsize,
        axes=dict(
            xlim=temperature_range,
            ylim=tuple(p / 1.0e9 for p in pressure_range),
            xlabel="Temperature (K)",
            ylabel="Pressure (GPa)",
            title="Pyrolite, 2% Fe³⁺/ΣFe on a no metal-basis",
        ),
        scope="SLB2024 solids; closed oxygen; high/low spin populations equilibrated; no liquids",
        notes="Hatched area: one or more phases have undefined equations of state. Numbered fields: ",
    )
    if key_path.exists():
        with key_path.open("a") as key:
            key.write("\n## Phase abbreviations\n\n| Label | Model |\n| --- | --- |\n")
            for label, name in PHASE_GLOSSARY.items():
                key.write(f"| {label} | {name} |\n")
            key.write(
                "\nA leading number means that many coexisting compositions of the same solution.\n"
            )
            key.write(
                f"\nDataset and bulk source: [Stixrude & Lithgow-Bertelloni (2024)]({PAPER_URL}).\n"
            )


if __name__ == "__main__":
    phase_diagram = bm.phase_diagram(
        PYROLITE_COMPOSITION.atomic_composition,
        PHASES,
        PRESSURE_RANGE,
        TEMPERATURE_RANGE,
        settings(SEEDS, verbose=True),
    )

    write_phase_diagram(
        phase_diagram,
        OUTPUT_DIR / "pyrolite_phase_diagram",
        save_json=save_json,
        plot=plot,
    )
