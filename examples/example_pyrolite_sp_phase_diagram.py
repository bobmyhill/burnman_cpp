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
2% Fe³⁺/ΣFe on a metal-free basis, calculated from 0-40 GPa in P-S coordinates.
It uses the Stixrude & Lithgow-Bertelloni (2024) thermodynamic dataset,
with a composition taken from Table 1 of that paper.

The example sets the entropy bounds as calculated from stable equilibria
at (0 GPa, 0 °C) and (140 GPa, 4000 °C).

A mask is applied to states where the full orthopyroxene model has
no stable EOS root or the calcium-ferrite-structured solution has negative
heat capacity.
"""

from functools import partial
from math import isfinite
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import (
    plot_phase_diagram_example,
    write_phase_diagram,
)

OUTPUT_DIR = Path(__file__).resolve().with_name("pyrolite_sp_phase_diagram_output")
SEEDS = 41
ENTROPY_UNIT = "kB/atom"  # Or "J/K" or "kJ/K".
PAPER_URL = "https://doi.org/10.1093/gji/ggae126"
MODEL_SET = model_sets.slb2024()
PHASE_GLOSSARY = MODEL_SET.phase_glossary
PHASES = MODEL_SET.phases

PRESSURE_RANGE = (0.0, 40.0e9)  # in Pa.
TEMPERATURE_RANGE = (273.15, 4273.15)  # in K.
ENTROPY_REFERENCES = ((0.0, 273.15), (140.0e9, 4273.15))

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


def settings(seeds=41, verbose=False):
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = opts.entropy_seeds = seeds
    # Mask the diagram where the EoS of Mg-Tschermak breaks down.
    opts.required_eos_phases = ["opx"]
    # Mask the diagram where endmembers of cf have negative heat capacities.
    opts.required_thermal_phases = ["cf"]
    opts.verbose = verbose
    return opts


def entropy_references(opts):
    """Return the two reference states' total entropies and their coordinates."""
    records = []
    for pressure, temperature in ENTROPY_REFERENCES:
        state = bm.stable_equilibrium(
            PYROLITE_COMPOSITION.atomic_composition,
            PHASES,
            pressure,
            temperature,
            opts,
        )
        if (
            not state.success
            or state.outside_model_domain
            or not isfinite(state.entropy)
        ):
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


def save_json(
    result,
    path,
    *,
    pressure_range=PRESSURE_RANGE,
    entropy_reference_states=(),
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
            display_entropy_unit=ENTROPY_UNIT,
            temperature_search_range_K=list(TEMPERATURE_RANGE),
            entropy_reference_states=list(entropy_reference_states),
        ),
    )


def plot(
    result,
    path,
    label_fontsize=7.0,
    *,
    pressure_range=PRESSURE_RANGE,
    entropy_unit=ENTROPY_UNIT,
    figsize=(12, 10),
):
    key_path = Path(path).with_name(Path(path).stem + "_assemblages.md")
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        pressure_unit="GPa",
        entropy_unit=entropy_unit,
        show_nodes=False,
        figsize=figsize,
        axes=dict(
            ylim=tuple(p / 1.0e9 for p in pressure_range),
            ylabel="Pressure (GPa)",
            title="Pyrolite, S-P phase diagram; 2% Fe³⁺/ΣFe on a no metal-basis",
        ),
        scope="SLB2024 solids; closed oxygen; high/low spin populations equilibrated; no liquids",
        notes="Hatched area: one or more phases have undefined equations of state. "
        "Entropy bounds: S(0 GPa, 0 °C) to S(140 GPa, 4000 °C). Numbered fields: ",
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
    opts = settings(SEEDS, verbose=True)
    references = entropy_references(opts)
    entropy_range = tuple(r["total_entropy_J_per_K"] for r in references)
    print(
        f"Entropy bounds: {entropy_range[0]:.8g}-{entropy_range[1]:.8g} J/K per 100 g.",
        flush=True,
    )
    phase_diagram = bm.phase_diagram(
        PYROLITE_COMPOSITION.atomic_composition,
        PHASES,
        diagram="PS",
        pressure_range=PRESSURE_RANGE,
        entropy_range=entropy_range,
        temperature_range=TEMPERATURE_RANGE,
        settings=opts,
    )

    write_phase_diagram(
        phase_diagram,
        OUTPUT_DIR / "pyrolite_sp_phase_diagram",
        save_json=partial(save_json, entropy_reference_states=references),
        plot=plot,
    )
