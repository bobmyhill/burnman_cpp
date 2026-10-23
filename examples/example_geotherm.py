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

"""Compare tabulated terrestrial geotherms, a material adiabat and a Mars profile.

Run with ``python examples/example_geotherm.py`` (requires matplotlib).
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import geotherm, seismic
from burnman_cpp.minerals import SLB11


def main():
    earth_depths = np.linspace(270e3, 2800e3, 250)
    profiles = [
        geotherm.BrownShankland(),
        geotherm.Anderson(),
        geotherm.StaceyContinental(),
        geotherm.StaceyOceanic(),
        geotherm.Katsura2022(),
        geotherm.Anzellini2013(),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    print("Temperatures at 1000 km depth:")
    for profile in profiles:
        temperatures = profile.temperatures(earth_depths)
        axes[0].plot(temperatures, earth_depths / 1e3, label=type(profile).__name__)
        print(f"{type(profile).__name__:20s} {profile.temperatures(1000e3):8.2f} K")

    anchor_depth, anchor_temperature = 270e3, 1600.0
    anchor_pressure = seismic.prem_model.pressure(anchor_depth)
    adiabat = geotherm.AdiabaticGeotherm(
        SLB11.periclase(), anchor_temperature, anchor_pressure
    )
    axes[0].plot(
        adiabat.temperatures(earth_depths),
        earth_depths / 1e3,
        "k--",
        label="Periclase adiabat",
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
