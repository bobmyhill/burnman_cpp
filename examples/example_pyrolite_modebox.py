#!/usr/bin/env python3
# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Pyrolite mineral modes along the SLB2024 Figure 13 isentrope.

Use the Table 1 cation ratios, initial Fe3+/total Fe = 0.02 and a 100 g bulk.
Find the entropy at 0 Pa and 1600 K, then hold it fixed to 135.751 GPa.
Oxygen stays closed; native iron and high/low-spin populations equilibrate.
Plot atomic fractions, summing coexisting copies of each solution model.
As in the paper's top panel, magnify native Fe 100 times downward from 1.
The horizontal axis is pressure rather than PREM depth; only solids are used.
Reference: Stixrude & Lithgow-Bertelloni (2024), doi:10.1093/gji/ggae126.

Run: python examples/example_pyrolite_modebox.py
Edit the settings below to change path sampling, progress output or destination.
"""

from pathlib import Path

import numpy as np

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24, load_dataset
from burnman_cpp.tools.phase_diagram_helpers import save_figure

N_POINTS = 801
VERBOSE = True
OUTPUT_DIR = Path(__file__).with_name("pyrolite_modebox_output")


def pyrolite_composition():
    """Preserve Table 1 cation ratios while setting 2% ferric iron and 100 g."""
    iron = bm.Composition(dict(FeO=8.05e-3, Fe2O3=0.18e-3), "mass")
    total_fe = iron.atomic_composition["Fe"]
    bulk = bm.Composition(
        dict(
            SiO2=45.0e-3,
            MgO=38.88e-3,
            CaO=3.18e-3,
            Al2O3=4.0e-3,
            Na2O=0.13e-3,
            Cr2O3=0.57e-3,
        ),
        "mass",
    )
    bulk.add_components(dict(FeO=0.98 * total_fe, Fe2O3=0.01 * total_fe), "molar")
    bulk.renormalize("mass", "total", 0.1)
    return bulk


def calculate():
    """Solve the surface reference and a linear P-S path with constant entropy."""
    bulk = pyrolite_composition()
    phases, _ = load_dataset(SLB24)
    for phase in phases:
        phase.set_name(SLB24.phase_aliases[phase.name])
    settings = bm.PhaseDiagramSettings()
    settings.verbose = VERBOSE
    surface = bm.stable_equilibrium(
        bulk.atomic_composition,
        phases,
        pressure=0.0,
        temperature=1600.0,
        settings=settings,
    )
    if not surface.success or surface.outside_model_domain:
        raise RuntimeError(surface.message)
    # Total S is in J/K for this 100 g bulk, not molar or specific entropy.
    # Each point searches all candidates, using its predecessor as a warm start.
    path = bm.calculate_equilibrium_path(
        bulk,
        phases,
        {"P": 0.0, "S": surface.entropy},
        {"P": 135.751e9, "S": surface.entropy},
        n_points=N_POINTS,
        settings=settings,
        temperature_range=(1400.0, 4500.0),
    )
    return path


def main():
    path = calculate()
    fractions = path.fractions("atomic")
    metals = [i for i, name in enumerate(path.phase_names) if name.startswith("Fe-")]
    native_iron = fractions[:, metals].sum(axis=1)
    pressure_GPa = path.coordinates("P") / 1e9
    figure, ax = bm.plot_mode_box(path, x=pressure_GPa, xlabel="Pressure (GPa)")
    ax.plot(
        pressure_GPa,
        1.0 - 100.0 * native_iron,
        "k--",
        label="Native Fe ×100 (downward from 1)",
    )
    ax.set(
        xlim=(0.0, 135.751),
        title="SLB2024 pyrolite: 1600 K surface isentrope; initial Fe³⁺/ΣFe = 0.02",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=8)
    figure.set_size_inches(12, 6)
    figure.set_layout_engine("constrained")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "pyrolite_modebox.npz",
        pressure=path.coordinates("P"),
        temperature=path.coordinates("T"),
        entropy=path.coordinates("S"),
        volume=path.coordinates("V"),
        phase_names=path.phase_names,
        atomic_fractions=fractions,
        molar_amounts=path.molar_amounts,
        native_iron_atomic_fraction=native_iron,
        success=path.success,
        diagnostics=np.asarray(path.diagnostics, dtype=str),
    )
    save_figure(figure, OUTPUT_DIR / "pyrolite_modebox.png", formats="svg")
    print(f"Solved {path.success.sum()}/{N_POINTS} points. Outputs: {OUTPUT_DIR}")
    for message in path.diagnostics:
        print(message)


if __name__ == "__main__":
    main()
