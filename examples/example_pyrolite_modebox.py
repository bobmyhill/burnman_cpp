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
Calculate and plot pyrolite mineral modes along the
Stixrude & Lithgow-Bertelloni (2024, doi:10.1093/gji/ggae126)
Figure 13 isentrope.

This example uses the composition from Table 1,
including an initial Fe3+/total Fe = 0.02 and a 100 g total mass.
The example first finds the entropy at 0 Pa and 1600 K,
then holds the entropy fixed to 136 GPa.

As in the Figure 13 top panel, the amount of native Fe is
magnified 100 times and plotted as a line downward from 1.
Depth is inferred from pressure using PREM.
"""

from pathlib import Path

import numpy as np

import burnman_cpp as bm
from burnman_cpp import seismic
from burnman_cpp.minerals import SLB_2024, load_dataset
from burnman_cpp.tools.phase_diagram_helpers import save_figure

N_POINTS = 1001
VERBOSE = True
OUTPUT_DIR = Path(__file__).with_name("pyrolite_modebox_output")


def pyrolite_composition():
    bulk = bm.Composition(
        dict(
            SiO2=45.0,
            MgO=38.88,
            FeO=8.05,
            Fe2O3=0.18,
            CaO=3.18,
            Al2O3=4.0,
            Na2O=0.13,
            Cr2O3=0.57,
        ),
        "mass",
    )
    bulk.renormalize("mass", "total", 0.1)
    return bulk


def calculate():
    """
    Solve the surface reference and a
    linear P-S path with constant entropy.
    """
    bulk = pyrolite_composition()
    phases, _ = load_dataset(SLB_2024)
    for phase in phases:
        phase.set_name(SLB_2024.phase_aliases[phase.name])
    settings = bm.PhaseDiagramSettings()
    settings.verbose = VERBOSE
    stable_equilibrium = bm.stable_equilibrium(
        bulk.atomic_composition,
        phases,
        pressure=0.0,
        temperature=1600.0,
        settings=settings,
    )
    if not stable_equilibrium.success or stable_equilibrium.outside_model_domain:
        raise RuntimeError(stable_equilibrium.message)
    # Total S is in J/K for this 100 g bulk,
    # not molar or specific entropy.
    # Each point searches all candidates,
    # using its predecessor as a warm start.
    path = bm.calculate_equilibrium_path(
        bulk,
        phases,
        {"P": 0.0, "S": stable_equilibrium.entropy},
        {"P": 136e9, "S": stable_equilibrium.entropy},
        n_points=N_POINTS,
        settings=settings,
        temperature_range=(1400.0, 4500.0),
    )
    return path


if __name__ == "__main__":
    path = calculate()
    fractions = path.fractions("atomic")
    metals = [i for i, name in enumerate(path.phase_names) if name.startswith("Fe-")]
    native_iron = fractions[:, metals].sum(axis=1)
    prem = seismic.PREM()
    depth = prem.depth(path.coordinates("P"))
    depth_km = depth / 1e3
    figure, ax = bm.plot_mode_box(path, x=depth_km, xlabel="Depth (km)")
    ax.plot(
        depth_km,
        1.0 - 100.0 * native_iron,
        "k--",
        label="Native Fe x100 (downward from 1)",
    )
    ax.set(
        xlim=prem.depth([0.0, 136e9]) / 1e3,
        ylim=(0.0, 1.0),
        title="SLB2024 pyrolite: 1600 K surface isentrope; initial Fe³⁺/ΣFe = 0.02",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=8)
    figure.set_size_inches(12, 6)
    figure.set_layout_engine("constrained")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "pyrolite_modebox.npz",
        pressure=path.coordinates("P"),
        depth=depth,
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
