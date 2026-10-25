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

"""Build a four-layer Earth-sized planet and compare its radial profiles with PREM.

Each layer has a prescribed material and temperature mode. The planet iterates
the equation of state, Poisson's equation and hydrostatic equilibrium to find
consistent pressure and gravity profiles. The iron cores use isothermal
compression laws at 300 K; the mantle includes a basal thermal boundary layer.
"""

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as burnman


def main():
    # Dewaele et al. (2006) solid iron, with the same Vinet parameters as BurnMan.
    inner_core = burnman.Layer("inner core", radii=np.linspace(0.0, 1220e3, 10))
    inner_core.set_material(
        burnman.Mineral(
            dict(
                name="Fe (Dewaele)",
                formula={"Fe": 1.0},
                equation_of_state="vinet",
                V_0=6.75e-6,
                K_0=163.4e9,
                Kprime_0=5.38,
                molar_mass=0.055845,
                n=1.0,
            )
        )
    )
    inner_core.set_temperature_mode("user-defined", np.full(inner_core.n_slices, 300.0))

    # Anderson & Ahrens (1994) liquid iron, using fourth-order Birch-Murnaghan.
    outer_core = burnman.Layer("outer core", radii=np.linspace(1220e3, 3480e3, 10))
    outer_core.set_material(
        burnman.Mineral(
            dict(
                name="Liquid Fe (Anderson)",
                formula={"Fe": 1.0},
                equation_of_state="bm4",
                V_0=7.95626e-6,
                K_0=109.7e9,
                Kprime_0=4.66,
                Kprime_prime_0=-0.043e-9,
                molar_mass=0.055845,
            )
        )
    )
    outer_core.set_temperature_mode("user-defined", np.full(outer_core.n_slices, 300.0))

    lower_mantle_radii = np.linspace(3480e3, 5711e3, 101)
    lower_mantle = burnman.Layer("lower mantle", radii=lower_mantle_radii)
    lower_mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
    # An 840 K basal perturbation and no upper thermal boundary perturbation.
    perturbation = burnman.BoundaryLayerPerturbation(
        radius_bottom=3480e3,
        radius_top=5711e3,
        rayleigh_number=1e7,
        temperature_change=840.0,
        boundary_layer_ratio=0.0,
    )
    lower_mantle.set_temperature_mode(
        "perturbed-adiabatic", temperatures=perturbation.temperature(lower_mantle_radii)
    )
    upper_mantle = burnman.Layer("upper mantle", radii=np.linspace(5711e3, 6371e3, 10))
    upper_mantle.set_material(burnman.minerals.SLB_2011.forsterite())
    upper_mantle.set_temperature_mode("adiabatic", temperature_top=1200.0)

    planet_zog = burnman.Planet(
        "Planet Zog", [inner_core, outer_core, lower_mantle, upper_mantle], verbose=True
    )
    print(planet_zog)
    planet_zog.make()
    print(
        f"\nmass/Earth= {planet_zog.mass/5.97e24:.3f}, "
        f"moment of inertia factor= {planet_zog.moment_of_inertia_factor:.4f}"
    )
    for layer in planet_zog:
        print(f"{layer.name} mass fraction of planet {layer.mass/planet_zog.mass:.3f}")

    prem = burnman.seismic.PREM()
    prem_radii = 6371e3 - prem.internal_depth_list()
    prem_density, prem_pressure, prem_gravity = prem.evaluate(
        ["density", "pressure", "gravity"]
    )
    figure, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    figure.suptitle(
        f"{planet_zog.name} has a mass {planet_zog.mass/5.97e24:.3f} times that of Earth,\n"
        f"an average density of {planet_zog.average_density:.1f} kg/m³,\n"
        f"and a moment of inertia factor of {planet_zog.moment_of_inertia_factor:.4f}",
        fontsize=12,
    )
    axes[0].plot(
        planet_zog.radii / 1e3,
        planet_zog.rho / 1e3,
        "k",
        linewidth=2,
        label=planet_zog.name,
    )
    axes[0].plot(prem_radii / 1e3, prem_density / 1e3, "--k", linewidth=1, label="PREM")
    axes[0].set_ylabel(r"Density ($10^3$ kg/m$^3$)")
    axes[0].set_ylim(0, planet_zog.rho.max() / 1e3 + 1)
    axes[0].legend()
    axes[1].plot(planet_zog.radii / 1e3, planet_zog.P / 1e9, "b", linewidth=2)
    axes[1].plot(prem_radii / 1e3, prem_pressure / 1e9, "--b", linewidth=1)
    axes[1].set_ylabel("Pressure (GPa)")
    axes[1].set_ylim(0, planet_zog.P.max() / 1e9 + 10)
    axes[2].plot(planet_zog.radii / 1e3, planet_zog.gravity, "g", linewidth=2)
    axes[2].plot(prem_radii / 1e3, prem_gravity, "--g", linewidth=1)
    axes[2].set_ylabel(r"Gravity (m/s$^2$)")
    axes[2].set_ylim(0, planet_zog.gravity.max() + 0.5)
    mask = planet_zog.T > 301.0
    axes[3].plot(planet_zog.radii[mask] / 1e3, planet_zog.T[mask], "r", linewidth=2)
    axes[3].set_ylabel("Temperature (K)")
    axes[3].set_xlabel("Radius (km)")
    axes[3].set_ylim(0, planet_zog.T.max() + 100)
    axes[3].set_xlim(0, planet_zog.radius_planet / 1e3)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    plt.show()
    return planet_zog, figure


if __name__ == "__main__":
    main()
