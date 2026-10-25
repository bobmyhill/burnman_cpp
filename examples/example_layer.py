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

"""Build a lower-mantle layer and compare its properties with PREM.

Pressures and gravity are calculated self-consistently for an adiabatic
perovskite-periclase mixture. PREM supplies the upper pressure and lower gravity
boundary conditions. All calculations use SI units.
"""

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as burnman
from burnman_cpp import minerals


def main():
    # Define the lower-mantle rock as 80 mol% perovskite and 20 mol% periclase.
    rock = burnman.Assemblage(
        [minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()], [0.8, 0.2]
    )
    seismic_model = burnman.seismic.PREM()
    depths = np.linspace(2890e3, 670e3, 20)
    pressure, gravity, seis_rho, seis_vp, seis_vs = seismic_model.evaluate(
        ["pressure", "gravity", "density", "v_p", "v_s"], depths
    )
    seis_bullen = seismic_model.bullen(depths)

    lower_mantle = burnman.Layer("Lower Mantle", radii=6371e3 - depths)
    lower_mantle.set_material(rock)
    lower_mantle.set_temperature_mode("adiabatic", temperature_top=1900.0)
    # To prescribe temperatures instead, use:
    # temperatures = burnman.geotherm.BrownShankland().temperatures(depths)
    # lower_mantle.set_temperature_mode("user-defined", temperatures)
    lower_mantle.set_pressure_mode(
        "self-consistent", pressure_top=pressure[-1], gravity_bottom=gravity[0]
    )
    # To prescribe pressures instead, use:
    # lower_mantle.set_pressure_mode("user-defined", pressure, gravity_bottom=gravity[0])
    lower_mantle.make()
    print(lower_mantle, end="")
    print(f"Layer mass: {lower_mantle.mass:.6e} kg")

    figure, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()
    for computed, reference, color, label in [
        (lower_mantle.v_p, seis_vp, "r", r"V$_P$"),
        (lower_mantle.v_s, seis_vs, "b", r"V$_S$"),
    ]:
        axes[0].plot(
            lower_mantle.P / 1e9,
            computed / 1e3,
            color + "-o",
            markersize=4,
            label=f"{label} (computed)",
        )
        axes[0].plot(
            pressure / 1e9,
            reference / 1e3,
            color + "--o",
            markerfacecolor="w",
            markersize=4,
            label=f"{label} (reference)",
        )
    axes[0].set_ylabel("Wave speeds (km/s)")
    axes[1].plot(
        lower_mantle.P / 1e9,
        lower_mantle.rho / 1e3,
        "g-o",
        markersize=4,
        label="computed",
    )
    axes[1].plot(
        pressure / 1e9,
        seis_rho / 1e3,
        "g--o",
        markerfacecolor="w",
        markersize=4,
        label="reference",
    )
    axes[1].set_ylabel(r"Density (g/cm$^3$)")
    axes[2].plot(
        lower_mantle.P / 1e9, lower_mantle.bullen, "k-o", markersize=4, label="computed"
    )
    axes[2].plot(
        pressure / 1e9,
        seis_bullen,
        "k--o",
        markerfacecolor="w",
        markersize=4,
        label="reference",
    )
    axes[2].set_ylabel("Bullen parameter")
    axes[3].plot(
        lower_mantle.P / 1e9, lower_mantle.T, "k-o", markersize=4, label="geotherm"
    )
    axes[3].set_ylabel("Temperature (K)")
    for axis in axes:
        axis.set_xlabel("Pressure (GPa)")
        axis.set_xlim(pressure.min() / 1e9, pressure.max() / 1e9)
        axis.legend(loc="best")
    figure.tight_layout()
    plt.show()
    return lower_mantle, figure


if __name__ == "__main__":
    main()
