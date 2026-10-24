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

"""Fit garnet compositions and phase proportions from compositional observations.

Fit molar abundances and ferric iron to the Jennings and Holland (2015) garnet
model and to a composition-only Solution class object. Then fit phase mass
fractions for the Bertka and Fei (1997) Mars mantle experiments. The phase
fraction error bars assume unit observation covariance for the six oxide
abundances, given in weight percent. All measurements are included here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import minerals
from burnman_cpp.optimize.composition_fitting import (
    DummyCompositionSolution,
    fit_composition_to_solution,
    fit_phase_proportions_to_bulk_composition,
)


def main():
    # Example 1: fit single-phase compositional data to a garnet model.
    gt = minerals.JH15.garnet()
    print(f"Endmembers: {gt.endmember_names}")
    print(f"Elements: {gt.elements}")
    print("Stoichiometric array:")
    print(gt.stoichiometric_matrix)

    fitted_species = ["Fe", "Ca", "Mg", "Cr", "Al", "Si", "Fe3+"]
    species_amounts = np.array([1.1, 2.0, 0.0, 0.0, 1.9, 3.0, 0.1])
    species_covariances = np.eye(7) * 0.01**2
    rng = np.random.RandomState(100)
    species_amounts = rng.multivariate_normal(species_amounts, species_covariances)

    print("\nObserved composition:")
    for species, amount, variance in zip(
        fitted_species, species_amounts, species_covariances.diagonal()
    ):
        print(f"{species}: {amount:.3f} +/- {np.sqrt(variance):.3f}")

    # Ferric iron occupies the second site, where it is labelled Fef.
    species_conversions = {"Fe3+": {"Fef_B": 1.0}}
    popt, pcov, res = fit_composition_to_solution(
        gt, fitted_species, species_amounts, species_covariances, species_conversions
    )
    gt.set_composition(popt)
    print("\nBest-fit molar fractions:")
    for name, fraction, variance in zip(
        gt.endmember_names, gt.molar_fractions, pcov.diagonal()
    ):
        print(f"{name}: {fraction:.3f} +/- {np.sqrt(variance):.3f}")

    # The same compositional fit is possible without a thermodynamic model.
    element_formulae = [
        "Mg3Al2Si3O12",
        "Fe3Al2Si3O12",
        "Ca3Al2Si3O12",
        "Ca3Fe2Si3O12",
        "Mg3Cr2Si3O12",
    ]
    site_formulae = [
        "[Mg]3[Al]2Si3O12",
        "[Fe]3[Al]2Si3O12",
        "[Ca]3[Al]2Si3O12",
        "[Ca]3[Fef]2Si3O12",
        "[Mg]3[Cr]2Si3O12",
    ]
    gt = DummyCompositionSolution(element_formulae, site_formulae)
    popt1, pcov1, res1 = fit_composition_to_solution(
        gt, fitted_species, species_amounts, species_covariances, species_conversions
    )
    np.testing.assert_allclose(popt1, popt, atol=1e-12)
    np.testing.assert_allclose(pcov1, pcov, atol=1e-12)
    np.testing.assert_allclose(res1, res, atol=1e-10)
    gt.set_composition(popt1)
    print("\nSite formula:")
    print(gt.site_formula(precision=2))
    print(f"\nWeighted residual: {res:.3f}\n")

    # Example 2: fit phase mass fractions to the Martian mantle bulk composition.
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
