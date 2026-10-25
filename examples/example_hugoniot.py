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

"""Compare equilibrium and non-reactive forsterite Hugoniots using SLB_2024.

The EquilibratedComposite class object selects the stable solid phases at
each pressure and temperature. Energy balance follows a fixed mass as the
number of formula-unit moles changes. Melting and transformation kinetics
are not included. All calculations use SI units.
The comparison retains forsterite as a single phase throughout compression.
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import EquilibratedComposite
from burnman_cpp.minerals import SLB_2024
from burnman_cpp.tools.chemistry import hugoniot


def main(material=None):
    """Plot the Hugoniot of a supplied EquilibratedComposite class object.

    Without an argument, use Mg2SiO4 and the Mg-Si-O endmembers in SLB_2024.
    """
    if material is None:
        bulk_composition = {"Mg": 2.0, "Si": 1.0, "O": 4.0}
        material = EquilibratedComposite(
            bulk_composition,
            SLB_2024,
            phase_names=[
                "forsterite",
                "mg_wadsleyite",
                "mg_ringwoodite",
                "mg_perovskite",
                "mg_postperovskite",
                "periclase",
                "mg_akimotoite",
                "mg_majorite",
                "enstatite",
                "clinoenstatite",
                "hp_clinoenstatite",
                "stishovite",
                "coesite",
                "quartz",
                "alpha_pbo_2_sio_2",
            ],
            name="Mg2SiO4 (SLB_2024)",
        )

    pressure_ref = 1.0e5
    temperature_ref = 300.0
    pressures = np.linspace(pressure_ref, 100.0e9, 201)
    material.set_state(pressure_ref, temperature_ref)
    reference = material.snapshot()
    temperatures = np.empty_like(pressures)

    # Phase formulae have different molar masses, so plot physical density
    # and mass fractions rather than interpreting a current mole as Mg2SiO4.
    densities = np.empty_like(pressures)
    phase_amounts = []
    names = []
    print("Pressure (GPa)  Temperature (K)  Density (kg/m³)  Phases")
    for i, pressure in enumerate(pressures):
        # Retain the energy-determined phase proportions at coexistence.
        # Each accepted state supplies the estimate for the following pressure.
        temperature, _ = hugoniot(
            material, pressure_ref, temperature_ref, [pressure], reference
        )
        temperatures[i] = temperature[0]
        densities[i] = material.density
        masses = material.phase_masses
        fractions = masses / masses.sum()
        amounts = {
            phase.name: fraction for phase, fraction in zip(material.phases, fractions)
        }
        phase_amounts.append(amounts)
        for name in amounts:
            if name not in names:
                names.append(name)
        print(
            f"{pressure/1.e9:14.2f}  {temperatures[i]:15.1f}  {densities[i]:16.1f}  "
            + ", ".join(amounts)
        )

    forsterite = SLB_2024.forsterite()
    forsterite_temperatures, forsterite_volumes = hugoniot(
        forsterite, pressure_ref, temperature_ref, pressures
    )
    forsterite_densities = forsterite.molar_mass / forsterite_volumes

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    figure.suptitle(f"Hugoniot: {material.name}")
    axes[0].plot(pressures / 1.0e9, temperatures, label="Equilibrium assemblage")
    axes[0].plot(
        pressures / 1.0e9,
        forsterite_temperatures,
        "--",
        label="Non-reactive forsterite",
    )
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend(fontsize=8)
    axes[1].plot(pressures / 1.0e9, densities / 1.0e3, label="Equilibrium assemblage")
    axes[1].plot(
        pressures / 1.0e9,
        forsterite_densities / 1.0e3,
        "--",
        label="Non-reactive forsterite",
    )
    axes[1].set_ylabel(r"Density (g/cm$^3$)")
    axes[1].legend(fontsize=8)
    axes[2].stackplot(
        pressures / 1.0e9,
        [[amounts.get(name, 0.0) for amounts in phase_amounts] for name in names],
        labels=names,
    )
    axes[2].set_ylabel("Phase mass fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="best", fontsize=8)
    for axis in axes:
        axis.set_xlabel("Pressure (GPa)")
        axis.set_xlim(pressures[0] / 1.0e9, pressures[-1] / 1.0e9)
    figure.tight_layout()
    plt.show()
    return material, figure


if __name__ == "__main__":
    main()
