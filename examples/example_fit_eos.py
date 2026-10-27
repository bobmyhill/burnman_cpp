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

"""Fit mineral equations of state, following Python example_fit_eos.py.

Fit room-temperature stishovite P-V data, then compare fits to periclase P-T-V
data alone and combined with low-pressure volume and enthalpy measurements.
The data, fit parameters and uncertainty assumptions follow the Python example.
The heat-capacity comparison shows how enthalpy constrains the thermal model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import Mineral, constants, eos_fitting, minerals, nonlinear_fitting

DATA_DIR = Path(__file__).resolve().parent / "data" / "input_fitting"


def molar_volume_from_unit_cell_volume(unit_cell_v, z):
    return unit_cell_v * constants.Avogadro / 1.0e30 / z


def print_fit(model):
    print("Optimized equation of state (SI units):")
    for name, value, variance in zip(model.fit_params, model.popt, np.diag(model.pcov)):
        print(f"{name}: {value:.8g} +/- {np.sqrt(variance):.3g}")
    print(f"Goodness of fit: {model.goodness_of_fit:.6g}\n")


def fit_stishovite():
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

    print_fit(fitted_eos)
    nonlinear_fitting.corner_plot(fitted_eos.popt, fitted_eos.pcov, params)
    plt.show()

    T = 298.15
    pressures = np.linspace(1.0e5, 60.0e9, 101)
    volumes = np.empty_like(pressures)
    for i, pressure in enumerate(pressures):
        stv.set_state(pressure, T)
        volumes[i] = stv.molar_volume

    fig, ax = plt.subplots()
    queries = np.column_stack([pressures, np.full(len(pressures), T), volumes])
    intervals = nonlinear_fitting.confidence_prediction_bands(
        fitted_eos,
        queries,
        0.95,
        lambda point: point[2],
        flag="V",
    )
    ax.fill_between(
        pressures / 1.0e9,
        intervals[2] * 1.0e6,
        intervals[3] * 1.0e6,
        color="C0",
        alpha=0.12,
        label="95% prediction",
    )
    ax.fill_between(
        pressures / 1.0e9,
        intervals[0] * 1.0e6,
        intervals[1] * 1.0e6,
        color="C0",
        alpha=0.3,
        label="95% confidence",
    )
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


def fit_periclase():
    print("2) Fitting to periclase high pressure PTV data only\n")

    # Dewaele et al. (2000): temperatures, pressures and unit-cell volumes.
    D2000_T, D2000_Terr, _, D2000_P, D2000_Perr, D2000_V, D2000_Verr = np.loadtxt(
        DATA_DIR / "PVT_MgO_Dewaele_et_al_2000.dat", unpack=True
    )
    D2000_P *= 1.0e9
    D2000_Perr *= 1.0e9
    D2000_V = molar_volume_from_unit_cell_volume(D2000_V, 4.0)
    D2000_Verr = molar_volume_from_unit_cell_volume(D2000_Verr, 4.0)

    PTV_data = np.column_stack([D2000_P, D2000_T, D2000_V])
    PTV_covariances = np.array(
        [
            np.diag(np.square(errors))
            for errors in zip(D2000_Perr, D2000_Terr, D2000_Verr)
        ]
    )
    # As in Python, fit at 0.1% fractional parameter tolerance initially.
    param_tolerance = 1.0e-3
    per_volume = minerals.SLB_2011.periclase()
    params = ["V_0", "K_0", "Kprime_0", "grueneisen_0", "q_0"]
    volume_fit = eos_fitting.fit_PTV_data(
        per_volume,
        params,
        PTV_data,
        PTV_covariances,
        param_tolerance=param_tolerance,
        verbose=False,
    )
    print_fit(volume_fit)
    nonlinear_fitting.corner_plot(volume_fit.popt, volume_fit.pcov, params)
    fig, ax = plt.subplots()
    nonlinear_fitting.weighted_residual_plot(
        ax, volume_fit, plot_axes=[0, 1], scale_axes=[1.0e-9, 1.0]
    )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Temperature (K)")
    plt.show()

    print("3) Fitting to periclase PTV and PT-enthalpy data\n")

    # Supplement the high-pressure data with Hazen (1976) lattice parameters.
    H1976_data = np.loadtxt(DATA_DIR / "Hazen_1976_TV_periclase.dat")
    H1976_T = H1976_data[:, 0] + 273.15
    H1976_Terr = np.abs(H1976_data[:, 0]) * 0.001
    H1976_P = np.full(len(H1976_data), 1.0e5)
    H1976_Perr = np.zeros(len(H1976_data))
    H1976_V = molar_volume_from_unit_cell_volume(H1976_data[:, 1] ** 3, 4.0)
    H1976_Verr = molar_volume_from_unit_cell_volume(
        3.0 * H1976_data[:, 1] ** 2 * H1976_data[:, 2], 4.0
    )
    PTV_data = np.column_stack(
        [
            np.concatenate([D2000_P, H1976_P]),
            np.concatenate([D2000_T, H1976_T]),
            np.concatenate([D2000_V, H1976_V]),
        ]
    )
    PTV_errors = np.column_stack(
        [
            np.concatenate([D2000_Perr, H1976_Perr]),
            np.concatenate([D2000_Terr, H1976_Terr]),
            np.concatenate([D2000_Verr, H1976_Verr]),
        ]
    )
    PTV_covariances = np.array([np.diag(errors**2) for errors in PTV_errors])

    # Victor and Douglas (1963): relative enthalpy in cal/mol, converted to J/mol.
    TH_data = np.loadtxt(DATA_DIR / "Victor_Douglas_1963_deltaH_MgO.dat")
    per_opt = minerals.SLB_2011.periclase()
    per_opt.set_state(1.0e5, 300.0)
    parameters = per_opt.params
    # Shift the initial energy reference as in Python; F_0 remains a fit variable.
    parameters.F_0 -= per_opt.molar_enthalpy
    per_opt = Mineral(parameters)
    PTH_data = np.column_stack(
        [
            np.full(len(TH_data), 1.0e5),
            TH_data[:, 0],
            TH_data[:, 2] * 4.184,
        ]
    )
    PTH_covariances = np.zeros((len(TH_data), 3, 3))
    PTH_covariances[:, 1, 1] = TH_data[:, 1]
    PTH_covariances[:, 2, 2] = (TH_data[:, 2] * 4.184 * 0.0004) ** 2

    # One property flag per row lets a single fit combine enthalpy and volume.
    flags = ["H"] * len(PTH_data) + ["V"] * len(PTV_data)
    PTp_data = np.concatenate([PTH_data, PTV_data])
    PTp_covariances = np.concatenate([PTH_covariances, PTV_covariances])
    fit_params = ["V_0", "K_0", "Kprime_0", "grueneisen_0", "q_0", "Debye_0", "F_0"]
    mixed_fit = eos_fitting.fit_PTp_data(
        mineral=per_opt,
        flags=flags,
        fit_params=fit_params,
        data=PTp_data,
        data_covariances=PTp_covariances,
        param_tolerance=param_tolerance,
        verbose=False,
    )
    print_fit(mixed_fit)
    nonlinear_fitting.corner_plot(mixed_fit.popt, mixed_fit.pcov, fit_params)
    fig, ax = plt.subplots()
    nonlinear_fitting.weighted_residual_plot(
        ax, mixed_fit, flag="V", plot_axes=[0, 1], scale_axes=[1.0e-9, 1.0]
    )
    ax.set_title("Weighted residual plot for volumes")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Temperature (K)")
    fig, ax = plt.subplots()
    nonlinear_fitting.plot_residuals(
        ax, mixed_fit.weighted_residuals, flags=mixed_fit.flags
    )
    plt.show()

    # Python's extreme-value check brackets the entire dataset at 90% confidence.
    confidence_interval = 0.9
    _, indices, probabilities = nonlinear_fitting.extreme_values(
        mixed_fit.weighted_residuals, confidence_interval
    )
    print(
        f"There are {len(indices)} outliers (at the "
        f"{100.0 * confidence_interval:.1f}% confidence interval). "
        "Their indices and probabilities are:"
    )
    for index, probability in zip(indices, probabilities):
        print(
            f"[{index}]: {100.0 * probability:.2f}% "
            f"({abs(mixed_fit.weighted_residuals[index]):.1f} s.d. from the model)"
        )
    print()

    # Compare the heat capacities predicted by the volume-only and joint fits.
    fig, ax = plt.subplots()
    for mineral, label, max_temperature, linestyle in (
        (minerals.HP_2011_ds62.per(), "HP", 2000.0, "--"),
        (minerals.SLB_2011.periclase(), "SLB", 2000.0, "--"),
        (per_volume, "PTV fit", 1200.0, "-"),
        (per_opt, "PTV + enthalpy fit", 2000.0, "-"),
    ):
        temperatures = np.linspace(200.0, max_temperature, 101)
        heat_capacities = np.empty_like(temperatures)
        for i, temperature in enumerate(temperatures):
            mineral.set_state(1.0e5, temperature)
            heat_capacities[i] = mineral.molar_heat_capacity_p
        ax.plot(temperatures, heat_capacities, linestyle=linestyle, label=label)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Heat capacity (J/K/mol)")
    ax.set_title("Periclase at 1 bar")
    ax.legend(loc="lower right")
    fig.tight_layout()

    # Plot enthalpy observations and the corresponding joint-fit prediction.
    fig, ax = plt.subplots()
    temperatures = np.linspace(298.15, 1200.0, 101)
    enthalpies = np.empty_like(temperatures)
    for i, temperature in enumerate(temperatures):
        per_opt.set_state(1.0e5, temperature)
        enthalpies[i] = per_opt.molar_enthalpy
    ax.plot(temperatures, enthalpies / 1.0e3, label="PTV + enthalpy fit")
    ax.errorbar(
        PTH_data[:, 1],
        PTH_data[:, 2] / 1.0e3,
        xerr=np.sqrt(PTH_covariances[:, 1, 1]),
        yerr=np.sqrt(PTH_covariances[:, 2, 2]) / 1.0e3,
        linestyle="None",
        marker="o",
        label="Victor and Douglas (1963)",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Relative enthalpy (kJ/mol)")
    ax.set_title("Periclase at 1 bar")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return volume_fit, mixed_fit


def main():
    print("Least squares equation of state fitting\n")
    stishovite_fit = fit_stishovite()
    volume_fit, mixed_fit = fit_periclase()
    return stishovite_fit, volume_fit, mixed_fit


if __name__ == "__main__":
    main()
