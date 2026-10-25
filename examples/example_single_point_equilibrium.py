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

"""Single-point phase discovery using any two of P, T, S and V.
Each call searches all candidate phases for a fixed bulk composition.
"""

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011


def main():
    """Recover one olivine-polymorph equilibrium using all six coordinate pairs."""
    bulk = bm.Composition(
        {"Mg2SiO4": 0.65, "Fe2SiO4": 0.35}, "molar"
    ).atomic_composition
    phases = [
        SLB_2011.mg_fe_olivine(),
        SLB_2011.mg_fe_wadsleyite(),
        SLB_2011.mg_fe_ringwoodite(),
    ]

    def point(**fixed):
        state = bm.stable_equilibrium(
            bulk,
            phases,
            **fixed,
            pressure_range=(12.0e9, 16.0e9),
            temperature_range=(1500.0, 1900.0),
        )
        if not state.success:
            raise RuntimeError(state.message)
        return state

    reference = point(pressure=14.0e9, temperature=1700.0)
    # Use this state's total S (J/K) and V (m^3) to demonstrate equivalent
    # constraints. In an application, supply the desired totals directly.
    p, t, s, v = (
        reference.pressure,
        reference.temperature,
        reference.entropy,
        reference.volume,
    )
    states = [
        ("P-T", reference),
        ("P-S", point(pressure=p, entropy=s)),
        ("P-V", point(pressure=p, volume=v)),
        ("T-S", point(temperature=t, entropy=s)),
        ("T-V", point(temperature=t, volume=v)),
        ("S-V", point(entropy=s, volume=v)),
    ]
    for pair, state in states:
        print(
            f"{pair}: P={state.pressure / 1.e9:.6g} GPa, "
            f"T={state.temperature:.6g} K, "
            f"S={state.entropy:.6g} J/K, V={state.volume * 1.e6:.6g} cm^3"
        )
        for phase in state.phases:
            print(
                f"  {phase.name}: {phase.amount:.6g} mol; "
                f"endmember fractions {phase.composition}"
            )


if __name__ == "__main__":
    main()
