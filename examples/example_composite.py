# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------


"""

example_composite
-----------------

This example demonstrates the functionalities of the burnman_cpp.Assemblage
class.

*Uses:*

* :ref:`ref-mineral-datasets`
* :class:`burnman_cpp.Mineral`
* :class:`burnman_cpp.Solution`
* :class:`burnman_cpp.Assemblage`

*Demonstrates:*

* How to initialize a composite object containing minerals and solutions
* How to set state and composition of composite objects
* How to interrogate composite objects for their
  compositional, thermodynamic and thermoelastic properties.
* How to use the stoichiometric and reaction affinity methods to solve
  simple thermodynamic equilibrium problems.

Please see example_equilibrate.py for more advanced (and user-friendly)
examples of equilibration of a composite material.
"""

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt


import burnman_cpp as burnman
from burnman_cpp import equilibrate
from burnman_cpp import minerals


def pretty_print_table(table):
    """Print the first column left aligned and the remaining columns right aligned."""
    rows = [[str(value) for value in row] for row in table]
    widths = [max(map(len, column)) + 1 for column in zip(*rows)]
    for row in rows:
        print(
            row[0].ljust(widths[0])
            + "".join(value.rjust(width) for value, width in zip(row[1:], widths[1:]))
        )


if __name__ == "__main__":
    print("Part I: Making and interrogating composite materials")

    # In BurnMan, composite materials are those made up of a mechanical
    # mixture of other materials. Those materials could be minerals,
    # solutions or other composites.

    # Initialising a composite material is easy; first initialise all the
    # material objects you want to add to the composite, and then call the
    # constructor for the Assemblage class.
    # The following lines do this for a mixture of bridgmanite,
    # ferropericlase and Ca-perovskite.
    bdg = burnman.minerals.SLB_2011.mg_fe_bridgmanite()
    fper = burnman.minerals.SLB_2011.ferropericlase()
    cpv = burnman.minerals.SLB_2011.ca_perovskite()

    rock = burnman.Assemblage(
        [bdg, fper, cpv],
        fractions=[0.7, 0.2, 0.1],
        fraction_type=burnman.FractionType.Molar,
    )
    rock.name = "lower mantle assemblage"

    # Assemblage takes explicit phase fractions; fraction_type defaults to molar.
    # Some methods are immediately available to us. For example, we can
    # print the endmembers of all the phases.

    print("Names of endmembers in composite material:")
    for name in rock.endmember_names:
        print(name)

    # We can also print a list of the potential elements in the composite:
    print("Elements which might be in the composite:")
    print(rock.elements)

    # and a stoichiometric array
    table = [rock.elements]
    table.extend([list(s) for s in rock.stoichiometric_matrix])

    print("Stoichiometric array:")
    pretty_print_table(table)

    # Before we can interrogate our new composite fully, we must also set
    # the compositions of any solutions, and also set the state of the
    # composite.
    bdg.set_composition([0.88, 0.07, 0.05])
    fper.set_composition([0.85, 0.15])
    rock.set_state(30.0e9, 2000.0)

    # Now we can print the rock:
    print("Composite composition")
    els = rock.elements  # Native element lists already follow IUPAC order.
    for e in els:
        print(f"{e}: {rock.formula[e]:.3f}")

    # and also many properties of the rock:
    print(f"Density: {rock.density:.3f} kg/m^3")
    print(f"Entropy: {rock.molar_entropy:.3f} J/K/mol")
    print(f"P-wave velocity: {rock.p_wave_velocity/1.e3:.3f} km/s")

    # We can also evaluate these properties at many pressures and
    # temperatures using the evaluate method:
    pressures = np.linspace(30.0e9, 130.0e9, 101)
    temperatures = 2000.0 * np.ones_like(pressures)
    Vp, Vs, rho = rock.evaluate(
        ["p_wave_velocity", "shear_wave_velocity", "density"], pressures, temperatures
    )
    plt.plot(pressures / 1.0e9, Vp / 1.0e3, label="V$_p$ (km/s)")
    plt.plot(pressures / 1.0e9, Vs / 1.0e3, label="V$_s$ (km/s)")
    plt.plot(pressures / 1.0e9, rho / 1.0e3, label="Density (g/cm$^3$)")
    plt.legend()
    plt.xlabel("Pressure (GPa)")
    plt.ylabel("Velocities, Density")
    plt.show()

    print("Part II: Manual equilibration")
    print("Example 1: Reaction in an andalusite-sillimanite rock\n")
    andalusite = minerals.HP_2011_ds62.andalusite()
    sillimanite = minerals.HP_2011_ds62.sill()
    kyanite = minerals.HP_2011_ds62.ky()

    rock = burnman.Assemblage([andalusite, sillimanite], [0.5, 0.5])
    rock.name = "and-sill assemblage"

    rock.set_state(1.0e9, 600.0)

    print("Isochemical reaction basis for rock")
    print(rock.reaction_basis_as_strings[0])

    print(
        f"Are we equilibrated at {rock.pressure/1.e9} GPa"
        f" and {rock.temperature} K? {rock.equilibrated}"
    )

    def delta_affinity_two_polymorphs(P, T, rock):
        rock.set_state(P, T)
        return rock.reaction_affinities[0]

    T = 600.0
    print(f"Equilibrating the rock with brentq at {int(T)} K")
    brentq(delta_affinity_two_polymorphs, 1.0e5, 1.0e9, args=(T, rock))

    print(f"Are we equilibrated now? {rock.equilibrated}")
    print(f"Equilibrium pressure: {rock.pressure/1.e9:.3f} GPa")

    print("\nExample 2: Reaction in an andalusite-sillimanite-kyanite rock\n")
    rock = burnman.Assemblage([andalusite, sillimanite, kyanite], [1.0 / 3.0] * 3)
    rock.name = "and-sill-ky assemblage"

    def delta_affinity_three_polymorphs(x):
        P, T = x
        rock.set_state(P, T)
        return rock.reaction_affinities

    print("Equilibrating the rock with fsolve")
    pt = fsolve(delta_affinity_three_polymorphs, [1.0e9, 1600.0])
    delta_affinity_three_polymorphs(pt)

    print(f"Are we equilibrated now? {rock.equilibrated}")
    print(f"Equilibrium pressure: {rock.pressure/1.e9:.3f} GPa")
    print(f"Equilibrium temperature: {rock.temperature:.3f} K")

    print("\nExample 3: Reaction in an olivine-wadsleyite rock\n")
    ol = burnman.minerals.SLB_2011.mg_fe_olivine()
    wad = burnman.minerals.SLB_2011.mg_fe_wadsleyite()

    rock = burnman.Assemblage([ol, wad], [0.5, 0.5])
    rock.name = "Olivine-wadsleyite assemblage"

    print(f"There are {rock.n_reactions} independent reactions in this rock.")
    print("One possible basis set is:")
    for string in rock.reaction_basis_as_strings:
        print(string)

    print("The reaction basis can also be returned as a numpy array:")
    print(rock.reaction_basis)

    print(
        f"If there are {rock.n_reactions} independent reactions, "
        f"we need to select {rock.n_reactions} unknowns."
    )
    print(
        "In this example, we find the pressure and composition \n"
        "of wadsleyite which coexists with fo90 at 1600 K.\n"
    )

    def delta_affinity_two_binaries_P_x_Fe_wad(x, T):
        P, x_fe_wad = x
        rock.set_state(P, T)
        wad.set_composition([1.0 - x_fe_wad, x_fe_wad])
        return rock.reaction_affinities

    print("Equilibrating the rock with fsolve")
    rock.set_fractions([1.0, 0.0])
    ol.set_composition([0.9, 0.1])
    px = fsolve(delta_affinity_two_binaries_P_x_Fe_wad, [13.0e9, 0.1], args=(1600.0))
    delta_affinity_two_binaries_P_x_Fe_wad(px, 1600.0)
    print(f"Are we equilibrated now? {rock.equilibrated}")
    print(f"{rock.name}: {rock.pressure/1.e9:.3f} GPa, {rock.temperature:.3f} K")
    for phase in rock.phases:
        print(f"  {phase.name}: {phase.molar_fractions}")

    print(
        "\nWe could alternatively choose to solve for the compositions of "
        "olivine and wadsleyite at fixed pressure and temperature"
    )

    # First, get the univariant pressure at the Mg-rich end of the binary
    # at a temperature of 1600 K.
    T = 1600.0
    fo_mwd = burnman.Assemblage(
        [
            burnman.minerals.SLB_2011.forsterite(),
            burnman.minerals.SLB_2011.mg_wadsleyite(),
        ],
        [0.5, 0.5],
    )
    brentq(delta_affinity_two_polymorphs, 12.0e9, 16.0e9, args=(T, fo_mwd))

    # Now solve for the equilibrium state at a range of pressures
    # less than the univariant pressure
    pressures = np.linspace(10.0e9, fo_mwd.pressure, 21)
    free_compositional_vectors = [{"Mg": 1.0, "Fe": -1.0}]
    composition = {"Mg": 1.8, "Fe": 0.2, "Si": 1.0, "O": 4.0}
    prm = burnman.get_equilibration_parameters(
        rock, composition, free_compositional_vectors
    )
    equality_constraints = [
        [burnman.PressureConstraint(pressure) for pressure in pressures[:-1]],
        burnman.TemperatureConstraint(T),
        burnman.PhaseFractionConstraint(0, 0.5, prm),
    ]
    # Set a positive amount of both phases before solving the coexistence grid.
    rock.set_fractions([0.5, 0.5])
    result = equilibrate(
        composition,
        rock,
        equality_constraints,
        free_compositional_vectors,
        # Pressure, temperature and composition use different SI scales.
        tol=np.r_[1.0, 1.0e-6, np.full(prm.n_parameters - 2, 1.0e-9)],
    )
    sol = result.sol_array.ravel()
    x_fe_ols = np.zeros(len(pressures))
    x_fe_wads = np.zeros(len(pressures))
    for i, solution in enumerate(sol):
        if not solution.success:
            raise RuntimeError(
                f"Equilibrium at {pressures[i]/1.e9:g} GPa: {solution.message}"
            )
        # Native results store parameter vectors rather than assemblage copies.
        burnman.set_composition_and_state_from_parameters(rock, solution.x)
        x_fe_ols[i] = rock.phases[0].molar_fractions[1]
        x_fe_wads[i] = rock.phases[1].molar_fractions[1]

    # The last pressure is the pure-Mg limit found with brentq above. There
    # is no Fe at this endpoint, so use the pure endmember equilibrium rather
    # than impose equality of the absent Fe endmembers' chemical potentials.
    ol.set_composition([1.0, 0.0])
    wad.set_composition([1.0, 0.0])
    rock.set_state(pressures[-1], T)

    # Pretty-print the results
    print(f"Olivine-wadsleyite equilibria at {T} K:")
    table = [["Pressure (GPa)", "p(Fe2SiO4, ol)", "p(Fe2SiO4, wad)"]]
    for i in range(len(pressures)):
        table.append(
            [f"{pressures[i]/1.e9:.3f}", f"{x_fe_ols[i]:.3f}", f"{x_fe_wads[i]:.3f}"]
        )
    pretty_print_table(table)

    # Plot the results
    plt.plot(x_fe_ols, pressures / 1.0e9, label="ol")
    plt.plot(x_fe_wads, pressures / 1.0e9, label="wad")
    plt.legend()
    plt.xlim(
        0,
    )
    plt.xlabel("p(Fe$_2$SiO$_4$)")
    plt.ylabel("Pressure (GPa)")
    plt.show()
