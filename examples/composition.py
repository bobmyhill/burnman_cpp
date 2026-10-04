"""Convert, normalize and change a bulk composition using only C++ calculations.

Run after installation: python examples/composition.py
"""

import burnman_cpp as bm


def main():
    composition = bm.Composition({"MgSiO3": 0.5, "FeSiO3": 0.5}, "molar")
    print("Initial elemental inventory (mol of atoms):", composition.atomic_composition)

    composition.renormalize("mass", "total", 1.0)
    composition.add_components({"Al2O3": 0.1}, "molar")
    atoms_before = composition.atomic_composition

    # Native nonnegative least squares expresses the same bulk in oxide components.
    composition.change_component_set(["MgO", "SiO2", "FeO", "Al2O3", "CaO"])
    composition.remove_null_components()
    composition.print("mass", significant_figures=5)
    composition.print("molar", significant_figures=3, normalization_amount=100.0)

    bulk = composition.atomic_composition
    for element, amount in atoms_before.items():
        assert abs(bulk[element] - amount) < 1.0e-10
    print("Bulk for equilibrate()/pseudosection():", bulk)


if __name__ == "__main__":
    main()
