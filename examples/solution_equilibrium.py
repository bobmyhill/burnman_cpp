"""Construct a regular oxide solution and equilibrate its bulk composition.

Run after installation: python examples/solution_equilibrium.py
The endmember parameters follow the repository's ferropericlase fixture;
property modifiers are omitted in this basic example.
"""

import burnman_cpp as bm


def main():
    periclase = bm.Mineral(
        {
            "name": "Periclase",
            "formula": {"Mg": 1.0, "O": 1.0},
            "equation_of_state": "slb3",
            "n": 2,
            "molar_mass": 0.0403044,
            "F_0": -569444.6,
            "V_0": 1.1244e-5,
            "K_0": 1.613836e11,
            "Kprime_0": 3.84045,
            "G_0": 1.309e11,
            "Gprime_0": 2.1438,
            "Debye_0": 767.0977,
            "grueneisen_0": 1.36127,
            "q_0": 1.7217,
            "eta_s_0": 2.81765,
        }
    )
    wuestite = bm.Mineral(
        {
            "name": "Wuestite",
            "formula": {"Fe": 1.0, "O": 1.0},
            "equation_of_state": "slb3",
            "n": 2,
            "molar_mass": 0.0718444,
            "F_0": -242146.0,
            "V_0": 1.2264e-5,
            "K_0": 1.794442e11,
            "Kprime_0": 4.9376,
            "G_0": 5.90e10,
            "Gprime_0": 1.44673,
            "Debye_0": 454.1592,
            "grueneisen_0": 1.53047,
            "q_0": 1.7217,
            "eta_s_0": -0.05731,
        }
    )
    model = bm.SymmetricRegularSolution(
        [(periclase, "[Mg]O"), (wuestite, "[Fe]O")],
        energy_interaction=[[13.0e3]],
    )
    oxide = bm.Solution(model, [0.8, 0.2], name="Ferropericlase")
    assemblage = bm.Assemblage([oxide], [1.0])
    result = bm.equilibrate(
        {"Mg": 0.6, "Fe": 0.4, "O": 1.0},
        assemblage,
        [bm.PressureConstraint(25.0e9), bm.TemperatureConstraint(2000.0)],
        store_iterates=True,
    )
    solve = result.sol_array[0, 0]
    if not solve.success:
        raise RuntimeError(solve.message)
    print(solve.message)
    print("Endmember fractions:", oxide.molar_fractions)
    print("Activities:", oxide.activities)
    print("Density (kg/m³):", assemblage.density)
    print("Residual norm:", solve.F_norm)


if __name__ == "__main__":
    main()
