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

"""Fit phase mass fractions for the Bertka and Fei (1997) Mars mantle experiments.

Fit the measured phase compositions, given in weight percent oxides, to the
reported bulk composition. Error bars assume unit observation covariance for
the six oxide abundances. All measurements are included with the example.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp.optimize.composition_fitting import (
    fit_phase_proportions_to_bulk_composition,
)


def main():
    filename = (
        Path(__file__).resolve().parent
        / "data/input_fitting/Bertka_Fei_1997_mars_mantle.dat"
    )
    data = np.genfromtxt(filename, names=True, dtype=None, encoding="utf8")

    # The first six columns are compositions given in weight percent oxides.
    compositions = np.array([data[name] for name in data.dtype.names[:6]])
    bulk_composition = compositions[:, 0]

    # Retain the order of experiments and phases in the measurements.
    samples = list(dict.fromkeys(data["sample"]))
    phases = list(dict.fromkeys(data["phase"]))
    samples.remove("bulk_composition")
    phases.remove("bulk")
    sample_indices = [np.flatnonzero(data["sample"] == sample) for sample in samples]
    pressures = np.array([data["pressure"][indices[0]] for indices in sample_indices])

    weight_proportions = np.full((len(samples), len(phases)), np.nan)
    weight_proportion_uncertainties = np.full_like(weight_proportions, np.nan)
    print("Experiment  Pressure (GPa)  Sum of squared residuals (wt %)^2")
    for i, sample in enumerate(samples):
        popt, pcov, res = fit_phase_proportions_to_bulk_composition(
            compositions[:, sample_indices[i]], bulk_composition
        )
        print(f"{sample:10s}  {pressures[i]:14.2f}  {res:35.6f}")

        sample_phases = data["phase"][sample_indices[i]]
        for j, phase in enumerate(phases):
            indices = np.flatnonzero(sample_phases == phase)
            if len(indices):
                # Combine multiple analyses of a mineral, including covariance.
                weight_proportions[i, j] = popt[indices].sum()
                weight_proportion_uncertainties[i, j] = np.sqrt(
                    pcov[np.ix_(indices, indices)].sum()
                )

    figure, axis = plt.subplots()
    for i, phase in enumerate(phases):
        axis.errorbar(
            pressures,
            weight_proportions[:, i],
            yerr=weight_proportion_uncertainties[:, i],
            fmt="o",
            label=phase,
            zorder=2,
        )

    axis.set(
        title="Phase proportions in the Martian mantle (Bertka and Fei, 1997)",
        xlim=(0.0, 40.0),
        ylim=(0.0, 1.0),
        xlabel="Pressure (GPa)",
        ylabel="Phase mass fraction",
    )
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure


if __name__ == "__main__":
    main()
