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

"""Fit stishovite pressure-volume data, following Python example_fit_eos.py.

This is the first section of Python BurnMan's example: the same Andrault et al.
(2003) data, stishovite model, pressure uncertainty and volume uncertainty.
It fits V_0, K_0 and Kprime_0, prints their uncertainties, and plots the fitted
isotherm and the maximum-likelihood data positions. All measurements are below.
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import constants, eos_fitting, minerals


def main():
    print("Least squares equation of state fitting\n")
    print("1) Fitting to stishovite room temperature PV data\n")
    # Pressure (GPa), unit-cell volume (angstrom^3), volume uncertainty (angstrom^3).
    PV = np.array(
        [
            [0.0001, 46.5126, 0.0061],
            [1.168, 46.3429, 0.0053],
            [2.299, 46.1756, 0.0043],
            [3.137, 46.0550, 0.0051],
            [4.252, 45.8969, 0.0045],
            [5.037, 45.7902, 0.0053],
            [5.851, 45.6721, 0.0038],
            [6.613, 45.5715, 0.0050],
            [7.504, 45.4536, 0.0041],
            [8.264, 45.3609, 0.0056],
            [9.635, 45.1885, 0.0042],
            [11.69, 44.947, 0.002],
            [17.67, 44.264, 0.002],
            [22.38, 43.776, 0.003],
            [29.38, 43.073, 0.009],
            [37.71, 42.278, 0.008],
            [46.03, 41.544, 0.017],
            [52.73, 40.999, 0.009],
            [26.32, 43.164, 0.006],
            [30.98, 42.772, 0.005],
            [34.21, 42.407, 0.003],
            [38.45, 42.093, 0.004],
            [43.37, 41.610, 0.004],
            [47.49, 41.280, 0.007],
        ]
    )

    Z = 2.0  # Formula units per stishovite unit cell.
    molar_volume_factor = constants.Avogadro * 1.0e-30 / Z
    PTV = np.column_stack(
        [PV[:, 0] * 1.0e9, np.full(len(PV), 298.15), PV[:, 1] * molar_volume_factor]
    )

    # Independent pressure and volume errors; temperature is exact.
    # As in the Python example, pressure uncertainty is 3% of total pressure.
    PTV_covariances = np.zeros((len(PV), 3, 3))
    PTV_covariances[:, 0, 0] = (0.03 * PTV[:, 0]) ** 2
    PTV_covariances[:, 2, 2] = (PV[:, 2] * molar_volume_factor) ** 2

    stv = minerals.HP_2011_ds62.stv()
    params = ["V_0", "K_0", "Kprime_0"]
    fitted_eos = eos_fitting.fit_PTV_data(
        stv, params, PTV, PTV_covariances, verbose=False
    )

    print("Optimized equation of state for stishovite (SI units):")
    for name, value, variance in zip(params, fitted_eos.popt, np.diag(fitted_eos.pcov)):
        print(f"{name}: {value:.8g} +/- {np.sqrt(variance):.3g}")
    print(f"Goodness of fit: {fitted_eos.goodness_of_fit:.6g}")

    T = 298.15
    pressures = np.linspace(1.0e5, 60.0e9, 101)
    volumes = np.empty_like(pressures)
    for i, pressure in enumerate(pressures):
        stv.set_state(pressure, T)
        volumes[i] = stv.molar_volume

    fig, ax = plt.subplots()
    ax.plot(pressures / 1.0e9, volumes * 1.0e6, label="Optimized fit for stishovite")
    ax.errorbar(
        PTV[:, 0] / 1.0e9,
        PTV[:, 2] * 1.0e6,
        xerr=np.sqrt(PTV_covariances[:, 0, 0]) / 1.0e9,
        yerr=np.sqrt(PTV_covariances[:, 2, 2]) * 1.0e6,
        linestyle="None",
        marker="o",
        label="Andrault et al. (2003)",
    )
    ax.plot(
        fitted_eos.data_mle[:, 0] / 1.0e9,
        fitted_eos.data_mle[:, 2] * 1.0e6,
        "k.",
        markersize=3,
        label="Maximum likelihood estimates",
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Volume (cm$^3$/mol)")
    ax.set_title("Stishovite at 298.15 K")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fitted_eos


if __name__ == "__main__":
    main()
