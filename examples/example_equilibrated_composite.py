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

"""Calculate equilibrium olivine polymorphs and their seismic velocities.

Compare relaxed velocities with unrelaxed velocities, which hold the
equilibrium phase fractions and compositions fixed during elastic deformation.

Run with ``python examples/example_equilibrated_composite.py``.
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import Composition, EquilibratedComposite
from burnman_cpp.minerals import SLB_2011


def main():
    # Uncomment to print all available phase_names in this dataset.
    # print(*EquilibratedComposite.available_phase_names(SLB_2011), sep="\n")

    bulk = Composition({"MgO": 1.8, "FeO": 0.2, "SiO2": 1.0}, "molar")
    rock = EquilibratedComposite(
        bulk,
        SLB_2011,
        phase_names=["mg_fe_olivine", "mg_fe_wadsleyite", "mg_fe_ringwoodite"],
        name="Equilibrated mantle rock",
    )
    pressures = np.linspace(1e9, 25e9, 40)
    temperatures = np.full_like(pressures, 1600.0)
    density, vp, vs = rock.evaluate(["rho", "v_p", "v_s"], pressures, temperatures)
    rock.relaxed = False
    vp_unrelaxed, vs_unrelaxed = rock.evaluate(["v_p", "v_s"], pressures, temperatures)
    rock.relaxed = True
    rock.set_state(18e9, 1600.0)
    print(rock)
    print("Mass-balance error:", rock.equilibrium_state.mass_balance_error)
    unrelaxed = rock.snapshot()
    print("Seismic velocities at 18 GPa and 1600 K (km/s):")
    print(f"Relaxed:   Vp = {rock.v_p / 1e3:.4f}, Vs = {rock.v_s / 1e3:.4f}")
    print(
        f"Unrelaxed: Vp = {unrelaxed.p_wave_velocity / 1e3:.4f}, "
        f"Vs = {unrelaxed.shear_wave_velocity / 1e3:.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].plot(density, pressures / 1e9)
    axes[0].set(xlabel="Density (kg/m³)", ylabel="Pressure (GPa)")
    axes[1].plot(vp / 1e3, pressures / 1e9, color="tab:blue", label="P wave (relaxed)")
    axes[1].plot(
        vs / 1e3, pressures / 1e9, color="tab:orange", label="S wave (relaxed)"
    )
    axes[1].plot(
        vp_unrelaxed / 1e3,
        pressures / 1e9,
        color="tab:blue",
        linestyle="--",
        label="P wave (unrelaxed)",
    )
    axes[1].plot(
        vs_unrelaxed / 1e3,
        pressures / 1e9,
        color="tab:orange",
        linestyle="--",
        label="S wave (unrelaxed)",
    )
    axes[1].set(xlabel="Velocity (km/s)")
    axes[1].legend()
    axes[0].invert_yaxis()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
