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

"""Plot the SPOCK pressure-volume reference isotherm for gold using C++.

Run after installation: python examples/example_spock_gold.py

Parameters: Myhill (2025), Table 3, https://doi.org/10.1093/gji/ggaf082,
using the less rounded Kdprime_0 from the paper's BurnMan companion script:
https://github.com/geodynamics/burnman/blob/main/contrib/spock_eos/Au_comparison.py
SPOCK has no temperature dependence; this curve represents the reference
isotherm fitted to the room-temperature data of Fratanduono et al. (2021).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as bm

GOLD_PARAMS = dict(
    name="Gold (Au)",
    formula={"Au": 1.0},
    equation_of_state="spock",
    molar_mass=0.19696657,
    V_0=10.1959e-6,
    K_0=156.6e9,
    Kprime_0=6.76,
    Kdprime_0=-15.6 / 156.6e9,
    Kprime_inf=2.97,
    P_0=1.0e5,
    F_0=0.0,
)


# Edit these settings directly before running the example.
MAX_PRESSURE_GPA = 1000.0
OUTPUT_DIR = Path(__file__).with_name("spock_gold_output")


def main():
    eos = bm.make_eos("spock")
    params = bm.mineral_params(GOLD_PARAMS)
    eos.validate_parameters(params)
    temperature = 298.15
    pressures = np.linspace(params.P_0, MAX_PRESSURE_GPA * 1.0e9, 801)
    volumes = np.array([eos.compute_volume(p, temperature, params) for p in pressures])
    relative_volumes = volumes / params.V_0
    if not np.all(np.diff(volumes) < 0.0):
        raise RuntimeError("Gold volumes must decrease with increasing pressure.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT_DIR / "gold_spock_pv.csv",
        np.column_stack((pressures / 1.0e9, volumes * 1.0e6, relative_volumes)),
        delimiter=",",
        header="pressure_GPa,molar_volume_cm3_per_mol,V_over_V0",
        comments="",
    )
    metadata = dict(
        parameters_SI=GOLD_PARAMS,
        temperature_K=temperature,
        source="https://doi.org/10.1093/gji/ggaf082",
        parameter_source=(
            "https://github.com/geodynamics/burnman/blob/main/"
            "contrib/spock_eos/Au_comparison.py"
        ),
    )
    (OUTPUT_DIR / "gold_spock_parameters.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.79, bottom=0.29)
    fig.text(0.12, 0.94, "Gold · SPOCK equation of state", fontsize=20, weight="bold")
    fig.text(0.12, 0.89, "298.15 K reference isotherm", color="#555555", fontsize=12)
    gold = "#ae7c12"
    ax.plot(volumes * 1.0e6, pressures / 1.0e9, color=gold, linewidth=2.8)
    ax.fill_between(volumes * 1.0e6, pressures / 1.0e9, color=gold, alpha=0.07)
    ax.set(
        xlabel=r"Molar volume (cm$^3$ mol$^{-1}$)",
        ylabel="Pressure (GPa)",
        xlim=(volumes[-1] * 1.0e6 * 0.97, params.V_0 * 1.0e6 * 1.02),
        ylim=(0.0, MAX_PRESSURE_GPA),
    )
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda v: v / (params.V_0 * 1.0e6),
            lambda relative: relative * params.V_0 * 1.0e6,
        ),
    )
    top.set_xlabel(r"Relative volume $V/V_0$", labelpad=9)
    top.spines["top"].set_visible(True)

    ax.text(
        0.97,
        0.92,
        "SPOCK · Myhill (2025)\nCalculated with BurnMan C++",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#665025",
        linespacing=1.7,
    )
    fig.text(
        0.12,
        0.12,
        r"$V_0 = 10.1959$ cm$^3$ mol$^{-1}$    $K_0 = 156.6$ GPa    $K'_0 = 6.76$"
        "\n"
        r"$K''_0 = -0.09962$ GPa$^{-1}$    $K'_\infty = 2.97$",
        fontsize=10,
        color="#444444",
        linespacing=1.7,
    )
    fig.text(
        0.12,
        0.035,
        "Parameters: Myhill (2025), Table 3 and BurnMan companion code.\n"
        "doi:10.1093/gji/ggaf082",
        fontsize=8.5,
        color="#666666",
        linespacing=1.5,
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"gold_spock_pv.{extension}", dpi=220)
    plt.close(fig)
    print(f"Saved gold SPOCK P–V plot and data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
