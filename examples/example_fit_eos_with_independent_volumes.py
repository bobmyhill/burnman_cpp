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

"""Fit pressure and Helmholtz energy using volume as an independent variable.

Following Python BurnMan's example_fit_eos_with_independent_volumes.py, fit
Andrault et al. (2003) stishovite data with HP and SLB equations of state, then
recover seven periclase parameters from noisy synthetic F(V,T) observations.
The measurements, uncertainties, initial guesses and random seed follow Python.
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import Mineral, constants, eos_fitting, minerals, nonlinear_fitting


def print_fit(model):
    for name, value, variance in zip(model.fit_params, model.popt, np.diag(model.pcov)):
        print(f"{name}: {value:.8g} +/- {np.sqrt(variance):.3g}")
    print(f"Goodness of fit: {model.goodness_of_fit:.6g}\n")


def fit_stishovite():
    print("1) Fitting to stishovite room temperature P(V) data\n")
    # Pressure (GPa), unit-cell volume (angstrom^3) and volume uncertainty.
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

    Z = 2.0
    volumes = PV[:, 1] * constants.Avogadro / 1.0e30 / Z
    volume_errors = PV[:, 2] * constants.Avogadro / 1.0e30 / Z
    VTP = np.column_stack([volumes, np.full(len(PV), 298.15), PV[:, 0] * 1.0e9])
    VTP_covariances = np.zeros((len(VTP), 3, 3))
    VTP_covariances[:, 0, 0] = volume_errors**2
    VTP_covariances[:, 2, 2] = (0.03 * VTP[:, 2]) ** 2

    params = ["V_0", "K_0", "Kprime_0"]
    stv = minerals.HP_2011_ds62.stv()
    hp_fit = eos_fitting.fit_VTP_data(stv, params, VTP, VTP_covariances, verbose=False)
    print("Optimized equation of state for stishovite (HP):")
    print_fit(hp_fit)

    # As in Python, remove the SLB property modifiers to allow direct P(V,T).
    stv_SLB = minerals.SLB_2011.stishovite()
    stv_SLB.set_property_modifiers([])
    slb_fit = eos_fitting.fit_VTP_data(
        stv_SLB, params, VTP, VTP_covariances, verbose=False
    )
    print("Optimized equation of state for stishovite (SLB):")
    print_fit(slb_fit)

    fig, ax = plt.subplots()
    plot_volumes = np.linspace(volumes.min() * 0.99, volumes.max(), 101)
    for model, label in ((hp_fit, "HP fit"), (slb_fit, "SLB fit")):
        pressures = np.array(
            [model.function([v, 298.15, 0.0], "P")[2] for v in plot_volumes]
        )
        if model is hp_fit:
            queries = np.column_stack(
                [plot_volumes, np.full(len(plot_volumes), 298.15), pressures]
            )
            intervals = nonlinear_fitting.confidence_prediction_bands(
                model,
                queries,
                0.95,
                lambda point: point[2],
                flag="P",
            )
            ax.fill_between(
                plot_volumes * 1.0e6,
                intervals[2] / 1.0e9,
                intervals[3] / 1.0e9,
                color="C0",
                alpha=0.12,
                label="95% prediction",
            )
            ax.fill_between(
                plot_volumes * 1.0e6,
                intervals[0] / 1.0e9,
                intervals[1] / 1.0e9,
                color="C0",
                alpha=0.3,
                label="95% confidence",
            )
        ax.plot(plot_volumes * 1.0e6, pressures / 1.0e9, label=label)
    ax.errorbar(
        VTP[:, 0] * 1.0e6,
        VTP[:, 2] / 1.0e9,
        xerr=np.sqrt(VTP_covariances[:, 0, 0]) * 1.0e6,
        yerr=np.sqrt(VTP_covariances[:, 2, 2]) / 1.0e9,
        linestyle="None",
        marker="o",
        label="Andrault et al. (2003)",
    )
    ax.set_xlabel("Volume (cm$^3$/mol)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title("Stishovite at 298.15 K")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return hp_fit, slb_fit


def fit_helmholtz():
    print("2) Fitting to synthetic F(V, T) data\n")
    per_SLB = minerals.SLB_2011.periclase()
    V_0 = per_SLB.params.V_0
    volumes = np.linspace(0.7 * V_0, V_0, 4)
    temperatures = np.linspace(500.0, 2000.0, 4)
    VV, TT = np.meshgrid(volumes, temperatures)
    V_data, T_data = VV.ravel(), TT.ravel()
    # Evaluate each (V,T) state directly; no array-evaluation helper is needed.
    F_data = np.empty(len(V_data))
    for i, (volume, temperature) in enumerate(zip(V_data, T_data)):
        per_SLB.set_state_with_volume(volume, temperature)
        F_data[i] = per_SLB.molar_helmholtz
    F_stdev = 10.0
    F_data += np.random.RandomState(100).normal(scale=F_stdev, size=F_data.shape)
    VTF_data = np.column_stack([V_data, T_data, F_data])
    VTF_covariances = np.zeros((len(F_data), 3, 3))
    VTF_covariances[:, 2, 2] = F_stdev**2
    flags = ["helmholtz"] * len(F_data)
    fit_params = ["V_0", "K_0", "Kprime_0", "grueneisen_0", "q_0", "Debye_0", "F_0"]

    parameters = per_SLB.params
    print("Actual equation of state (SI units):")
    for name in fit_params:
        field = "debye_0" if name == "Debye_0" else name
        print(f"{name}: {getattr(parameters, field):.4e}")
    print()
    # Use the same deliberately displaced initial guesses as Python.
    parameters.F_0 += 2300.0
    parameters.V_0 *= 0.9
    parameters.K_0 *= 0.9
    per_SLB = Mineral(parameters)
    model = eos_fitting.fit_VTp_data(
        mineral=per_SLB,
        flags=flags,
        fit_params=fit_params,
        data=VTF_data,
        data_covariances=VTF_covariances,
        verbose=False,
    )
    print("Optimized equation of state (SI units):")
    print_fit(model)

    fig, ax = plt.subplots()
    for temperature in temperatures:
        mask = T_data == temperature
        curve_volumes = np.linspace(volumes[0], volumes[-1], 101)
        energies = np.array(
            [
                model.function([volume, temperature, 0.0], "helmholtz")[2]
                for volume in curve_volumes
            ]
        )
        (line,) = ax.plot(
            curve_volumes * 1.0e6, energies / 1.0e3, label=f"{temperature:.0f} K"
        )
        queries = np.column_stack(
            [curve_volumes, np.full(len(curve_volumes), temperature), energies]
        )
        intervals = nonlinear_fitting.confidence_prediction_bands(
            model,
            queries,
            0.95,
            lambda point: point[2],
            flag="helmholtz",
        )
        ax.fill_between(
            curve_volumes * 1.0e6,
            intervals[2] / 1.0e3,
            intervals[3] / 1.0e3,
            color=line.get_color(),
            alpha=0.12,
        )
        ax.fill_between(
            curve_volumes * 1.0e6,
            intervals[0] / 1.0e3,
            intervals[1] / 1.0e3,
            color=line.get_color(),
            alpha=0.3,
        )
        ax.errorbar(
            V_data[mask] * 1.0e6,
            F_data[mask] / 1.0e3,
            yerr=F_stdev / 1.0e3,
            color=line.get_color(),
            marker="o",
            linestyle="None",
        )
    ax.set_xlabel("Volume (cm$^3$/mol)")
    ax.set_ylabel("Helmholtz energy (kJ/mol)")
    ax.set_title("Synthetic periclase data and fitted isotherms")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return model


def main():
    print("Least squares equation of state fitting with independent volumes.\n")
    hp_fit, slb_fit = fit_stishovite()
    helmholtz_fit = fit_helmholtz()
    return hp_fit, slb_fit, helmholtz_fit


if __name__ == "__main__":
    main()
