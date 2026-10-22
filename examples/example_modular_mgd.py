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

"""Compare SLB fayalite with additive anharmonic modular MGD.

Based on BurnMan's examples/example_modular_mgd.py. This example starts
with native SLB_2011 fayalite, selects a BM3 reference and SLB Debye model,
and adds a power-law volume prefactor and log-normal thermal contribution.
The figures compare seven properties along isobars and isochores.

Run: python examples/example_modular_mgd.py
Edit the settings below; this script takes no command-line arguments.
All calculations use burnman_cpp; Python BurnMan is not required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import burnman_cpp as bm
from burnman_cpp import (
    anharmonic_prefactor_models,
    anharmonic_thermal_models,
    debye_temperature_models,
)
from burnman_cpp.minerals.SLB11 import fayalite

# Settings follow the original BurnMan example.
TEMPERATURES = np.linspace(10.0, 5000.0, 101)
PRESSURES = np.linspace(0.0, 30.0e9, 7)
RELATIVE_VOLUMES = np.linspace(1.2, 0.8, 5)
OUTPUT_DIR = Path(__file__).with_name("modular_mgd_output")
SHOW_PLOTS = True

PROPERTIES = (
    "molar_volume",
    "molar_heat_capacity_v",
    "molar_heat_capacity_p",
    "thermal_expansivity",
    "grueneisen_parameter",
    "isothermal_bulk_modulus_reuss",
)


def make_minerals():
    # params is an independent, editable snapshot of the native SLB mineral.
    m_SLB = fayalite()
    params = m_SLB.params
    params.equation_of_state = bm.EOSType.ModularMGD
    params.reference_eos = bm.make_eos("bm3")
    params.debye_temperature_model = debye_temperature_models.SLB()

    # A(V) = a_anh * (V/V_0)^m_anh multiplies the thermal contribution.
    params.anharmonic_prefactor_model = anharmonic_prefactor_models.PowerLaw()
    params.a_anh = 1.0
    params.m_anh = 10.0
    params.anharmonic_thermal_model = anharmonic_thermal_models.LogNormal()
    params.mu_anh = 0.0
    params.sigma_anh = 1.0
    m = bm.Mineral(params)

    # Preserve SLB_2011 fayalite's constant entropy correction, as in BurnMan.
    m.set_property_modifiers(
        [("linear", dict(delta_E=0.0, delta_S=26.76, delta_V=0.0))]
    )
    return m_SLB, m


def print_properties(mineral):
    mineral.set_state(5.0e9, 4000.0)
    print(f"Properties of {mineral.name} at 5.0 GPa and 4000 K:")
    print(f"Volume: {mineral.molar_volume * 1.0e6:.2f} cm³/mol")
    print(f"Entropy: {mineral.molar_entropy:.2f} J/(mol K)")
    print(f"Isochoric heat capacity: {mineral.molar_heat_capacity_v:.2f} J/(mol K)")
    print(f"Isobaric heat capacity: {mineral.molar_heat_capacity_p:.2f} J/(mol K)")
    print(f"Thermal expansivity: {mineral.thermal_expansivity * 1.0e6:.2f} /MK")
    print(f"Grüneisen parameter: {mineral.grueneisen_parameter:.3f}")
    print(
        f"Isothermal bulk modulus: {mineral.isothermal_bulk_modulus_reuss / 1.0e9:.2f} GPa"
    )
    print(
        f"Isentropic bulk modulus: {mineral.isentropic_bulk_modulus_reuss / 1.0e9:.2f} GPa"
    )


def evaluate_isobar(mineral, pressure, temperatures):
    """Evaluate the stable branch, leaving unavailable states as NaN."""
    values = np.full((len(PROPERTIES), len(temperatures)), np.nan)
    for i, temperature in enumerate(temperatures):
        try:
            mineral.set_state(pressure, temperature)
            column = [getattr(mineral, name) for name in PROPERTIES]
        except (ValueError, RuntimeError):
            # The source example also truncates curves outside the stable
            # EOS domain. Keep the actual temperature grid instead of guessing
            # a replacement upper temperature.
            continue
        if np.all(np.isfinite(column)):
            values[:, i] = column
    return values


def evaluate_isochore(mineral, volume, temperatures):
    """Evaluate P(V,T) and properties directly, without another V(P,T) solve.

    The source minerals' modifiers do not change the six plotted properties:
    fayalite has only a constant entropy offset, and periclase has none.
    """
    params = mineral.params
    eos = bm.make_eos(params.equation_of_state)
    methods = (
        eos.compute_molar_heat_capacity_v,
        eos.compute_molar_heat_capacity_p,
        eos.compute_thermal_expansivity,
        eos.compute_grueneisen_parameter,
        eos.compute_isothermal_bulk_modulus_reuss,
    )
    values = np.full((len(PROPERTIES), len(temperatures)), np.nan)
    for i, temperature in enumerate(temperatures):
        try:
            pressure = eos.compute_pressure(temperature, volume, params)
            column = [pressure] + [
                method(pressure, temperature, volume, params) for method in methods
            ]
        except (ValueError, RuntimeError):
            continue
        if np.all(np.isfinite(column)):
            values[:, i] = column
    return values


def plot_comparison(m_SLB, m, temperatures, paths, *, fixed_volume=False):
    """Draw the source example's seven thermodynamic-property panels."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    axes = axes.ravel()
    axes[-1].set_axis_off()
    relative_volume = m_SLB.params.V_0
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(paths)))
    for path, color in zip(paths, colors):
        if fixed_volume:
            volume = path * relative_volume
            reference = evaluate_isochore(m_SLB, volume, temperatures)
            modified = evaluate_isochore(m, volume, temperatures)
            label = rf"$V/V_0 = {path:.1f}$"
        else:
            reference = evaluate_isobar(m_SLB, path, temperatures)
            modified = evaluate_isobar(m, path, temperatures)
            label = f"{path / 1.0e9:g} GPa"

        for values, style in ((reference, ":"), (modified, "-")):
            first, cv, cp, alpha, gamma, kt = values
            scaled = (
                first / 1.0e9 if fixed_volume else first / relative_volume,
                cv,
                cp,
                alpha * 1.0e5,
                gamma,
                kt / 1.0e9,
                alpha * kt / 1.0e6,
            )
            for ax, y in zip(axes, scaled):
                ax.plot(temperatures, y, color=color, linestyle=style, linewidth=1.6)
        axes[1].plot([], [], color=color, label=label)

    labels = (
        "$P$ (GPa)" if fixed_volume else "$V/V_0$",
        "$C_V$ (J/mol/K)",
        "$C_P$ (J/mol/K)",
        r"$\alpha$ ($10^{-5}$/K)",
        r"$\gamma$",
        "$K_T$ (GPa)",
        r"$\alpha K_T$ (MPa/K)",
    )
    for ax, label in zip(axes, labels):
        ax.set(xlabel="Temperature (K)", ylabel=label, xlim=(0.0, temperatures[-1]))
        ax.grid(alpha=0.15)
    axes[1].legend(fontsize=8, frameon=False)
    axes[-1].legend(
        handles=[
            Line2D([], [], color="black", linestyle=":", label="SLB reference"),
            Line2D([], [], color="black", linestyle="-", label="Modular MGD"),
        ],
        loc="center",
        frameon=False,
    )
    kind = "fixed volume" if fixed_volume else "fixed pressure"
    fig.suptitle(f"{m.name}: SLB and modular MGD at {kind}")
    fig.tight_layout()
    return fig, axes


def main():
    m_SLB, m = make_minerals()
    print_properties(m)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plot_comparison(m_SLB, m, TEMPERATURES, PRESSURES)
    axes[1].set_ylim(0.0, 180.0)
    axes[2].set_ylim(0.0, 250.0)
    axes[3].set_ylim(0.0, 30.0)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fayalite_isobars.{extension}", dpi=180)

    fig, axes = plot_comparison(
        m_SLB, m, TEMPERATURES, RELATIVE_VOLUMES, fixed_volume=True
    )
    axes[2].set_ylim(0.0, 400.0)
    axes[3].set_ylim(0.0, 30.0)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"fayalite_isochores.{extension}", dpi=180)
    print(f"Saved figures to {OUTPUT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
