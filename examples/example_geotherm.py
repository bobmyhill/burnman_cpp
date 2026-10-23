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

"""Compare Earth geotherms, periclase and pyrolite adiabats, and a Mars profile.

Both adiabats use a potential temperature of 1620 K at 0 GPa.
Run with ``python examples/example_geotherm.py`` (requires matplotlib).
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import Composition, EquilibratedComposite, geotherm
from burnman_cpp.minerals import SLB24

# Dry pyrolite: relative oxide masses in kg.
PYROLITE_COMPOSITION = Composition(
    dict(
        SiO2=45.0e-3,
        MgO=38.88e-3,
        FeO=8.05e-3,
        Fe2O3=0.18e-3,
        CaO=3.18e-3,
        Al2O3=4.0e-3,
        Na2O=0.13e-3,
        Cr2O3=0.57e-3,
    ),
    "mass",
)
PYROLITE_COMPOSITION.renormalize("mass", "total", 0.1)

POTENTIAL_TEMPERATURE = 1620.0
REFERENCE_PRESSURE = 0.0


def main():
    earth_depths = np.linspace(270e3, 2800e3, 200)
    profiles = [
        geotherm.BrownShankland(),
        geotherm.Anderson(),
        geotherm.StaceyContinental(),
        geotherm.StaceyOceanic(),
        geotherm.Katsura2022(),
        geotherm.Anzellini2013(),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    print(
        f"Adiabats: potential temperature {POTENTIAL_TEMPERATURE:.0f} K "
        f"at {REFERENCE_PRESSURE / 1e9:.0f} GPa."
    )
    print("Temperatures at 1000 km depth:")
    for profile in profiles:
        temperatures = profile.temperatures(earth_depths)
        axes[0].plot(temperatures, earth_depths / 1e3, label=type(profile).__name__)
        print(f"{type(profile).__name__:20s} {profile.temperatures(1000e3):8.2f} K")

    adiabat = geotherm.AdiabaticGeotherm(
        SLB24.periclase(), POTENTIAL_TEMPERATURE, REFERENCE_PRESSURE
    )
    axes[0].plot(
        adiabat.temperatures(earth_depths),
        earth_depths / 1e3,
        "k--",
        label="Periclase adiabat",
    )

    earth_depths = np.linspace(270e3, 2800e3, 12)
    pyrolite = EquilibratedComposite(PYROLITE_COMPOSITION, SLB24, name="Pyrolite")
    pyrolite_adiabat = geotherm.AdiabaticGeotherm(
        pyrolite, POTENTIAL_TEMPERATURE, REFERENCE_PRESSURE
    )
    # Each temperature calculation allows phases and their compositions to change.
    print("Calculating the equilibrated pyrolite adiabat...", flush=True)
    pyrolite_temperatures = pyrolite_adiabat.temperatures(earth_depths)
    axes[0].plot(
        pyrolite_temperatures,
        earth_depths / 1e3,
        marker="o",
        color="tab:red",
        linewidth=2.0,
        label="Equilibrated pyrolite adiabat (SLB24, 12 points)",
    )
    print(
        f"{'Equilibrated pyrolite':20s} {pyrolite_adiabat.temperatures(1000e3):8.2f} K"
    )
    axes[0].set(title="Earth", xlabel="Temperature (K)", ylabel="Depth (km)")
    axes[0].legend(fontsize=8)

    mars = geotherm.Plesa2022Mars6cm3()
    mars_depths = np.linspace(mars.minimum_depth, mars.maximum_depth, 200)
    axes[1].plot(mars.temperatures(mars_depths), mars_depths / 1e3)
    axes[1].set(
        title="Mars: Plesa (2022), 6 cm³/mol",
        xlabel="Temperature (K)",
        ylabel="Depth (km)",
    )
    for axis in axes:
        axis.invert_yaxis()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
