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

# Adapted from BurnMan's examples/example_equilibrate.py.
# Copyright (C) 2012–2021 by the BurnMan team, GPL v2 or later.
# This adaptation is distributed under GPL v3 or later.
"""
example_equilibrate
--------------------

This example demonstrates how BurnMan may be used to calculate the
equilibrium state for an assemblage of a fixed bulk composition given
two constraints. Each constraint is a native PressureConstraint,
TemperatureConstraint, EntropyConstraint, VolumeConstraint, LinearXConstraint,
PTEllipseConstraint, PhaseFractionConstraint or PhaseCompositionConstraint.
A list of constraint objects samples a range of values; multiple lists
produce a Cartesian grid of equilibrium states.

Phase-fraction and phase-composition constraints take the index of the phase
in the assemblage and its equilibration parameters. A phase-composition
constraint also specifies site names, numerator and denominator coefficients,
and the desired ratio of site occupancies. For example, the constraint
([Mg_A, Fe_A], [1., 0.], [1., 1.], 0.5) would correspond to equal amounts
of Mg and Fe on the A site.

The native solver returns an object containing sol_array and prm. Each
solution contains the equilibrium parameter vector x; phase amounts and
endmember proportions can be read from that vector using prm.parameter_names.

This script provides a number of examples, which can be turned on and off
with a series of boolean variables. In order of complexity:

* run_aluminosilicates: Creates the classic aluminosilicate diagram involving
  univariate reactions between andalusite, sillimanite and kyanite.
* run_ordering: Calculates the state of order of Jennings and Holland (2015)
  orthopyroxene in the simple en-fs binary at 1 bar.
* run_gt_solvus: Demonstrates the shape of the pyrope-grossular solvus.
* run_fper_ol: Calculates the equilibrium Mg-Fe partitioning between
  ferropericlase and olivine.
* run_fixed_ol_composition: Calculates the composition of wadsleyite in
  equilibrium with olivine of a fixed composition at a fixed pressure.
* run_upper_mantle: Calculates the equilibrium compositions and
  phase proportions for an ol-opx-gt composite in an NCFMAS bulk composition.
* run_lower_mantle: Calculates temperatures and assemblage properties
  along an isentrope in the lower mantle. Includes calculations of the
  post-perovskite-in and bridgmanite-out lines.
* run_olivine_polymorphs: Produces a P-T pseudosection for a fo90 composition.

*Uses:*

* :ref:`ref-mineral-datasets`
* :class:`burnman_cpp.Assemblage`
* :func:`burnman_cpp.equilibrate`
"""

import numpy as np
import matplotlib.pyplot as plt

import burnman_cpp
from burnman_cpp import equilibrate
from burnman_cpp.minerals import HP_2011_ds62, SLB_2011, JH_2015
from burnman_cpp.tools.polytope import simplify_composite_with_composition


def print_assemblage(assemblage):
    print(f"P, T: {assemblage.pressure:.4g} Pa, {assemblage.temperature:.4g} K")
    print("Phase and endmember fractions:")
    for phase, fraction in zip(assemblage.phases, assemblage.molar_fractions):
        print(f"  {phase.name}: {fraction:.4f}")
        if isinstance(phase, burnman_cpp.Solution):
            for name, value in zip(phase.endmember_names, phase.molar_fractions):
                print(f"    {name}: {value:.4f}")


run_aluminosilicates = True
run_ordering = True
run_gt_solvus = True
run_fper_ol = True
run_fixed_ol_composition = True
run_upper_mantle = True
run_lower_mantle = True
run_olivine_polymorphs = True

gt = SLB_2011.garnet()
ol = SLB_2011.mg_fe_olivine()
wad = SLB_2011.mg_fe_wadsleyite()
rw = SLB_2011.mg_fe_ringwoodite()
bdg = SLB_2011.mg_fe_bridgmanite()
ppv = SLB_2011.post_perovskite()
fper = SLB_2011.ferropericlase()
opx = SLB_2011.orthopyroxene()
stv = SLB_2011.stishovite()
coe = SLB_2011.coesite()
cpv = SLB_2011.ca_perovskite()

if __name__ == "__main__" and run_aluminosilicates:
    """
    Creates the classic aluminosilicate diagram involving
    univariate reactions between andalusite, sillimanite and kyanite.
    """
    sillimanite = HP_2011_ds62.sill()
    andalusite = HP_2011_ds62.andalusite()
    kyanite = HP_2011_ds62.ky()

    # First, find the pressure and temperature of the invariant point
    composition = sillimanite.formula
    assemblage = burnman_cpp.Assemblage(
        [sillimanite, andalusite, kyanite], [1.0 / 3.0] * 3
    )
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.PhaseFractionConstraint(2, 0.0, prm),
        burnman_cpp.PhaseFractionConstraint(0, 0.0, prm),
    ]

    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm
    P_inv, T_inv = sol.x[0:2]
    print(f"invariant point found at {P_inv/1e9:.2f} GPa, {T_inv:.2f} K")

    # Now we can find the univariant lines which all converge on the
    # invariant point. In this case, we assume we know which side of each line
    # is stable (because the aluminosilicate diagram is so ubiquitous in
    # metamorphic textbooks), but if we didn't know,
    # we could also calculate which field is stable around the invariant point
    # by checking to see which had the minimum Gibbs energy.
    low_pressures = np.linspace(1.0e5, P_inv, 21)
    high_pressures = np.linspace(P_inv, 1.0e9, 21)

    for pair, pressures in [
        ([andalusite, kyanite], low_pressures),
        ([sillimanite, andalusite], low_pressures),
        ([sillimanite, kyanite], high_pressures),
    ]:
        assemblage = burnman_cpp.Assemblage(pair, [0.5, 0.5])
        prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
        equality_constraints = [
            [burnman_cpp.PressureConstraint(P) for P in pressures],
            burnman_cpp.PhaseFractionConstraint(0, 0.0, prm),
        ]
        result = equilibrate(composition, assemblage, equality_constraints)
        sols, prm = result.sol_array.ravel(), result.prm
        Ps = np.array([sol.x[0] for sol in sols if sol.success])
        Ts = np.array([sol.x[1] for sol in sols if sol.success])
        plt.plot(Ts, Ps / 1.0e9)

    # Now we plot each stable field. Here, we do explicitly check
    # which polymorph is stable at each point.
    assemblage = burnman_cpp.Assemblage(
        [sillimanite, andalusite, kyanite], [1.0 / 3.0] * 3
    )
    for P, T in [[0.2e9, 800.0], [0.6e9, 650], [0.5e9, 1000]]:
        assemblage.set_state(P, T)

        stable_phase = kyanite
        min_gibbs = kyanite.molar_gibbs

        for phase in [sillimanite, andalusite]:
            if phase.molar_gibbs < min_gibbs:
                min_gibbs = phase.molar_gibbs
                stable_phase = phase

        plt.text(T, P / 1.0e9, stable_phase.name)

    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")

    plt.show()

if __name__ == "__main__" and run_ordering:
    """
    Calculates the state of order of Jennings and Holland (2015)
    orthopyroxene in the simple en-fs binary at 1 bar.
    """
    assemblage = burnman_cpp.Assemblage([JH_2015.orthopyroxene()], [1.0])

    composition = {
        "Mg": 50 / 100.0 * 2.0,
        "Fe": (1.0 - 50 / 100.0) * 2.0,
        "Si": 2.0,
        "O": 6.0,
    }

    # The JH2015 orthopyroxene has some components which do not
    # exist in our simple bulk composition, so here we call a
    # helper function to simplify any phases that cannot be represented
    # by the bulk composition. This can either remove endmembers, or in some
    # cases redefine the independent set of endmembers to create the simplest
    # solutions possible.
    assemblage = simplify_composite_with_composition(assemblage, composition)

    assemblage.phases[0].set_composition([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    temperatures = np.linspace(2000.0, 300.0, 41)

    Mg_numbers = np.linspace(10.0, 50.0, 5)
    equality_constraints = [
        burnman_cpp.PressureConstraint(1.0e5),
        [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
    ]

    # Loop over Mg numbers
    for Mg_number in Mg_numbers:
        composition = {
            "Mg": Mg_number / 100.0 * 2.0,
            "Fe": (1.0 - Mg_number / 100.0) * 2.0,
            "Si": 2.0,
            "O": 6.0,
        }
        result = equilibrate(composition, assemblage, equality_constraints)
        sols, prm = result.sol_array.ravel(), result.prm
        # We know that we've made a simpler orthopyroxene solution.
        # In this case, the endmembers are either the same, or
        # the processing has created an anti-ordered endmember.
        # We get the index of the ordered/anti-ordered endmember here.
        try:
            idx = prm.parameter_names.index(
                "p(ordered ferroenstatite in orthopyroxene (CFMASCrO) (transformed))"
            )
            p_fms = np.array([sol.x[idx] for sol in sols if sol.success])
        except ValueError:
            idx = prm.parameter_names.index(
                "p(Derived member (occupancies: [Mg][Fe][Si]1/2) in orthopyroxene (CFMASCrO) (transformed))"
            )
            p_fms = np.array([-sol.x[idx] for sol in sols if sol.success])

        Ts = np.array([sol.x[1] for sol in sols if sol.success])
        plt.plot(Ts, p_fms, label=f"Mg# = {Mg_number}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Proportion of ordered orthopyroxene")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__" and run_gt_solvus:
    """
    Demonstrates the shape of the pyrope-grossular solvus.
    """
    composition = {"Mg": 1.5, "Ca": 1.5, "Al": 2.0, "Si": 3.0, "O": 12.0}

    # The assemblage in this case is made of two distinct copies of the
    # SLB_2011 garnet solution.
    # In the bulk composition of interest, there are only
    # two stable endmembers of the solution. Here, we simplify each phase
    # so that it only contains those endmembers
    a = burnman_cpp.Assemblage([SLB_2011.garnet()], [1.0])
    a = simplify_composite_with_composition(a, composition)
    py_gr = a.phases[0]

    assemblage = burnman_cpp.Assemblage([py_gr, py_gr.copy()], [0.5, 0.5])

    # We must set the compositions as distinct so that the equilibrium
    # function approaches the minimum with two distinct compositions,
    # rather than the metastable equilibrium where both phases have
    # the same composition.
    assemblage.phases[0].set_composition([0.05, 0.95])
    assemblage.phases[1].set_composition([0.95, 0.05])

    # Run the equilibration
    pressure = 1.0e5
    temperatures = np.linspace(300.0, 601.4, 301)
    equality_constraints = [
        burnman_cpp.PressureConstraint(pressure),
        [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
    ]

    result = equilibrate(composition, assemblage, equality_constraints)
    sols, prm = result.sol_array.ravel(), result.prm

    # Interrogate the stable assemblages for phase compositions.
    # Binary solutions store their second endmember fraction in x.
    x1s = np.array([sol.x[3] for sol in sols if sol.success])
    x2s = np.array([sol.x[5] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])

    plt.plot(x1s, Ts)
    plt.plot(x2s, Ts)

    plt.text(0.5, 400.0, "miscibility gap", horizontalalignment="center")
    plt.xlabel("Molar proportion of pyrope")
    plt.ylabel("Temperature (K)")
    plt.show()

if __name__ == "__main__" and run_fper_ol:
    """
    Calculates the equilibrium Mg-Fe partitioning between
    ferropericlase and olivine.
    """
    temperatures = np.linspace(1000.0, 1500.0, 11)
    pressures = np.linspace(0.0e9, 10.0e9, 3)

    assemblage = burnman_cpp.Assemblage([ol, fper], [0.7, 0.3])

    ol.set_composition([0.93, 0.07])
    fper.set_composition([0.9, 0.1])

    assemblage.set_state(pressures[0], temperatures[0])
    equality_constraints = [
        [burnman_cpp.PressureConstraint(P) for P in pressures],
        [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
    ]

    composition = {"Mg": 1.0, "Fe": 0.5, "Si": 0.5, "O": 2.5}

    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array, result.prm

    for sols in sol:
        PGPa = sols[0].x[0] / 1.0e9
        T = [s.x[1] for s in sols]
        x_fa = [s.x[3] for s in sols]
        x_wus = [s.x[5] for s in sols]
        p = plt.plot(T, x_fa, label=f"p(fa), {PGPa} GPa")
        plt.plot(
            T, x_wus, c=p[0].get_color(), linestyle="--", label=f"p(wus), {PGPa} GPa"
        )
    plt.xlabel("Temperature (K)")
    plt.ylabel("proportion (mol fraction)")
    plt.legend()
    plt.show()

if __name__ == "__main__" and run_fixed_ol_composition:
    """
    Calculates the composition of wadsleyite in equilibrium with olivine
    of a fixed composition at a fixed pressure.
    """
    assemblage = burnman_cpp.Assemblage([ol, wad], [0.7, 0.3])
    ol.set_composition([0.5, 0.5])
    wad.set_composition([0.6, 0.4])

    assemblage.set_state(10.0e9, 1200.0)
    composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.PressureConstraint(10.0e9),
        burnman_cpp.PhaseCompositionConstraint(
            0, ["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45, assemblage, prm
        ),
    ]

    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm

    print_assemblage(assemblage)

if __name__ == "__main__" and run_upper_mantle:
    """
    Calculates the equilibrium compositions and phase proportions for
    an ol-opx-gt composite in an NCFMAS bulk composition.
    """
    pressure = 10.0e9
    temperature = 1500.0

    composition = {
        "Na": 0.02,
        "Fe": 0.2,
        "Mg": 2.0,
        "Si": 1.9,
        "Ca": 0.2,
        "Al": 0.4,
        "O": 6.81,
    }

    assemblage = burnman_cpp.Assemblage([ol, opx, gt], [0.7, 0.1, 0.2])

    ol.set_composition([0.93, 0.07])
    opx.set_composition([0.8, 0.1, 0.05, 0.05])
    gt.set_composition([0.8, 0.1, 0.05, 0.03, 0.02])

    assemblage.set_state(pressure, temperature)
    equality_constraints = [
        burnman_cpp.PressureConstraint(10.0e9),
        burnman_cpp.TemperatureConstraint(1500.0),
    ]

    result = equilibrate(
        composition, assemblage, equality_constraints, max_iterations=20
    )
    sol, prm = result.sol_array.item(), result.prm
    print_assemblage(assemblage)

if __name__ == "__main__" and run_lower_mantle:
    """
    Calculates temperatures and assemblage properties along an isentrope
    in the lower mantle. Includes calculations of the
    post-perovskite-in and bridgmanite-out lines.
    """
    P0 = 25.0e9
    T0 = 1600.0

    bdg.name = "bdg"
    ppv.name = "ppv"
    fper.name = "fper"
    cpv.name = "cpv"

    bdg.set_composition([0.86, 0.1, 0.04])
    ppv.set_composition([0.86, 0.1, 0.04])
    fper.set_composition([0.9, 0.1])

    composition = {"Fe": 0.2, "Mg": 2.0, "Si": 1.9, "Ca": 0.2, "Al": 0.4, "O": 6.8}
    assemblage = burnman_cpp.Assemblage([bdg, fper, cpv], [1.0 / 3.0] * 3)

    # We start by calculating the assemblage entropy at a reference
    # pressure and temperature.
    equality_constraints = [
        burnman_cpp.PressureConstraint(P0),
        burnman_cpp.TemperatureConstraint(T0),
    ]
    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm

    S = assemblage.n_moles * assemblage.molar_entropy

    # Now we calculate the assemblages and states at the post-perovskite-in and
    # bridgmanite-out boundaries at the desired entropy.
    assemblage = burnman_cpp.Assemblage([bdg, fper, ppv, cpv], [0.25] * 4)
    assemblage.set_state(sol.x[0], sol.x[1])
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.EntropyConstraint(S),
        burnman_cpp.PhaseFractionConstraint(2, 0.0, prm),
    ]
    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm
    P_ppv_in = assemblage.pressure
    T_ppv_in = assemblage.temperature

    assemblage.set_state(sol.x[0], sol.x[1])
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.EntropyConstraint(S),
        burnman_cpp.PhaseFractionConstraint(0, 0.0, prm),
    ]
    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm
    P_bdg_in = assemblage.pressure
    T_bdg_in = assemblage.temperature

    # Now we calculate the properties of each assemblage
    # along its stable pressure segments.
    prp_list = []
    prm_list = []
    for pressures, phases in [
        [np.linspace(25.0e9, P_ppv_in, 21), [bdg, fper, cpv]],
        [np.linspace(P_ppv_in, P_bdg_in, 21), [bdg, fper, ppv, cpv]],
        [np.linspace(P_bdg_in, 140.0e9, 21), [ppv, fper, cpv]],
    ]:
        assemblage = burnman_cpp.Assemblage(
            phases, np.full(len(phases), 1.0 / len(phases))
        )
        equality_constraints = [
            [burnman_cpp.PressureConstraint(P) for P in pressures],
            burnman_cpp.EntropyConstraint(S),
        ]
        result = equilibrate(composition, assemblage, equality_constraints)
        sols, prm = result.sol_array.ravel(), result.prm
        prm_list.append(prm)
        prp_list.append(np.array([sol.x for sol in sols if sol.success]).T)

    # Finally, do some plotting
    fig = plt.figure(figsize=(16, 8))
    ax = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
    phases = [bdg, fper, cpv, ppv]
    colors = ["red", "green", "blue", "orange", "purple"]
    for j in range(3):
        prm = prm_list[j]
        p = prp_list[j]
        pressures, temperatures = p[[0, 1], :]
        ax[0].plot(pressures / 1.0e9, temperatures, color="black")
        ax[0].set_xlabel("Pressure (GPa)")
        ax[0].set_ylabel("Temperature (K)")
        for i, phase in enumerate(phases):
            try:
                idx = prm.parameter_names.index(f"x({phase.name})")
                x_phase = p[idx, :]
                ax[i + 1].plot(pressures / 1.0e9, x_phase, color=colors[i])
                ax[i + 1].set_ylim(0, 2)
                ax[i + 1].set_xlim(0, 140)
                ax[i + 1].set_xlabel("Pressure (GPa)")
                ax[i + 1].set_ylabel(f"n moles of {phase.name}")
            except ValueError:
                pass

        if (
            f"x({phases[0].name})" in prm.parameter_names
            and f"x({phases[1].name})" in prm.parameter_names
        ):
            pv_idx = prm.parameter_names.index(f"x({phases[0].name})")
            per_idx = prm.parameter_names.index(f"x({phases[1].name})")
            KD = (
                p[pv_idx + 1, :]
                * (1.0 - p[per_idx + 1, :])
                / ((1.0 - p[pv_idx + 1, :] - p[pv_idx + 2, :]) * p[per_idx + 1, :])
            )
            if j == 1:
                ax[5].plot(pressures / 1.0e9, KD, color="red", label="bdg K$_D$")
            else:
                ax[5].plot(pressures / 1.0e9, KD, color="red")

        if (
            f"x({phases[1].name})" in prm.parameter_names
            and f"x({phases[3].name})" in prm.parameter_names
        ):
            per_idx = prm.parameter_names.index(f"x({phases[1].name})")
            ppv_idx = prm.parameter_names.index(f"x({phases[3].name})")
            KD = (
                p[ppv_idx + 1, :]
                * (1.0 - p[per_idx + 1, :])
                / ((1.0 - p[ppv_idx + 1, :] - p[ppv_idx + 2, :]) * p[per_idx + 1, :])
            )
            if j == 1:
                ax[5].plot(pressures / 1.0e9, KD, color="blue", label="ppv K$_D$")
            else:
                ax[5].plot(pressures / 1.0e9, KD, color="blue")

    ax[5].set_ylim(0.0, 1.0)
    ax[5].set_xlabel("Pressure (GPa)")
    ax[5].set_ylabel("[FeSiO$_3$/MgSiO$_3$]/[FeO/MgO]")
    ax[5].legend()

    plt.show()

if __name__ == "__main__" and run_olivine_polymorphs:
    """
    Produces a P-T pseudosection for a fo90 composition.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    T0 = 573.15
    T1 = 1773.15

    ol.set_composition([0.93, 0.07])
    wad.set_composition([0.91, 0.09])
    rw.set_composition([0.93, 0.07])
    bdg.set_composition([0.86, 0.1, 0.04])
    fper.set_composition([0.9, 0.1])

    plt.text(1750.0, 10.0, "olivine", horizontalalignment="right")
    plt.text(1750.0, 15.0, "wadsleyite", horizontalalignment="right")
    plt.text(1750.0, 20.0, "ringwoodite", horizontalalignment="right")
    plt.text(1750.0, 24.0, "bridgmanite + ferropericlase", horizontalalignment="right")

    composition = {"Fe": 0.2, "Mg": 1.8, "Si": 1.0, "O": 4.0}
    assemblage = burnman_cpp.Assemblage([ol, wad, rw], [1.0, 0.0, 0.0])
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.PhaseFractionConstraint(0, 0.0, prm),
        burnman_cpp.PhaseFractionConstraint(2, 0.0, prm),
    ]

    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm
    Pinv1, Tinv1 = sol.x[0:2]

    if not sol.success:
        raise Exception("Couldn't find ol-wad-rw invariant")

    assemblage = burnman_cpp.Assemblage([ol, wad, rw], [1.0 / 3.0] * 3)
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        burnman_cpp.PhaseFractionConstraint(1, 0.0, prm),
        burnman_cpp.PhaseFractionConstraint(2, 0.0, prm),
    ]
    result = equilibrate(composition, assemblage, equality_constraints)
    sol, prm = result.sol_array.item(), result.prm
    Pinv2, Tinv2 = sol.x[0:2]

    for d in [
        [np.linspace(T0, Tinv1, 8), [ol, wad, rw], ol, 0.0],
        [np.linspace(T0, Tinv2, 8), [ol, wad, rw], wad, 0.0],
        [np.linspace(Tinv2, Tinv1, 8), [ol, wad, rw], rw, 0.0],
        [np.linspace(T0, Tinv2, 8), [ol, rw], rw, 0.0],
        [np.linspace(Tinv2, T1, 8), [ol, wad], wad, 0.0],
        [np.linspace(Tinv1, T1, 8), [ol, wad], wad, 1.0],
        [np.linspace(Tinv1, T1, 8), [wad, rw], rw, 0.0],
    ]:
        temperatures, phases, phase, fraction = d

        assemblage = burnman_cpp.Assemblage(
            phases, np.full(len(phases), 1.0 / len(phases))
        )
        prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
        equality_constraints = [
            [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
            burnman_cpp.PhaseFractionConstraint(phases.index(phase), fraction, prm),
        ]
        result = equilibrate(composition, assemblage, equality_constraints)
        sols, prm = result.sol_array.ravel(), result.prm
        Ps = np.array([sol.x[0] for sol in sols if sol.success])
        Ts = np.array([sol.x[1] for sol in sols if sol.success])
        ax.plot(Ts, Ps / 1.0e9, color="k")

    assemblage.set_fractions([0.0, 1.0])
    temperatures = np.linspace(T0, T1, 8)
    prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
    equality_constraints = [
        [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
        burnman_cpp.PhaseFractionConstraint(1, 1.0, prm),
    ]
    result = equilibrate(composition, assemblage, equality_constraints)
    sols, prm = result.sol_array.ravel(), result.prm
    Ps = np.array([sol.x[0] for sol in sols if sol.success])
    Ts = np.array([sol.x[1] for sol in sols if sol.success])
    ax.plot(Ts, Ps / 1.0e9, color="k")

    assemblage = burnman_cpp.Assemblage([rw, bdg, fper], [1.0 / 3.0] * 3)
    assemblage = simplify_composite_with_composition(assemblage, composition)

    assemblage.phases[1].set_composition([0.86, 0.14])

    for phase_fraction in [0.0, 1.0]:
        prm = burnman_cpp.get_equilibration_parameters(assemblage, composition)
        equality_constraints = [
            [burnman_cpp.TemperatureConstraint(T) for T in temperatures],
            burnman_cpp.PhaseFractionConstraint(0, phase_fraction, prm),
        ]
        result = equilibrate(composition, assemblage, equality_constraints)
        sols, prm = result.sol_array.ravel(), result.prm
        Ps = np.array([sol.x[0] for sol in sols if sol.success])
        Ts = np.array([sol.x[1] for sol in sols if sol.success])
        ax.plot(Ts, Ps / 1.0e9, color="k")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_ylim(5.0, 25.0)
    plt.show()
