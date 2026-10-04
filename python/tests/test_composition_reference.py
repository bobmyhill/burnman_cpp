"""Parity with the main Python BurnMan Composition API (optional dependency)."""

import pytest
import burnman_cpp as bm

pytest.importorskip("burnman")
from burnman.classes.composition import Composition as ReferenceComposition
from burnman.utils.chemistry import atomic_masses


def compare(native, pure):
    for basis in ("mass", "weight", "molar", "atomic"):
        assert native.composition(basis) == pytest.approx(
            dict(pure.composition(basis)), rel=2e-12, abs=1e-15
        )
    assert native.component_formulae == pure.component_formulae


@pytest.mark.parametrize(
    "data",
    [
        {"MgO": 1.0, "SiO2": 1.0},
        {"FeSiO3": 0.5, "MgSiO3": 0.5, "Al2O3": 0.2},
        {
            "SiO2": 0.52,
            "Al2O3": 0.16,
            "MgO": 0.08,
            "FeO": 0.08,
            "CaO": 0.10,
            "Na2O": 0.035,
            "K2O": 0.005,
            "H2O": 0.02,
        },
        {"Mg1/2Fe0.5SiO3": 2.0, "CH3COOH": 0.1},
    ],
)
@pytest.mark.parametrize("unit", ["mass", "weight", "molar"])
@pytest.mark.parametrize("normalize", [False, True])
def test_input_conversions_against_python(data, unit, normalize):
    native = bm.Composition(data, unit, normalize)
    pure = ReferenceComposition(data, unit, normalize)
    compare(native, pure)
    assert list(native.mass_composition) == list(pure.mass_composition)
    assert native.element_list == pure.element_list


@pytest.mark.parametrize("element", sorted(set(atomic_masses) - {"Vc"}))
def test_complete_atomic_mass_table_against_python(element):
    native = bm.Composition({element: 1.0}, "molar")
    pure = ReferenceComposition({element: 1.0}, "molar")
    compare(native, pure)
    assert native.mass_composition[element] == atomic_masses[element]


@pytest.mark.parametrize(
    "unit,component,target",
    [
        ("mass", "total", 1.0),
        ("weight", "MgO", 0.2),
        ("molar", "total", 10.0),
        ("molar", "SiO2", 2.0),
        ("atomic", "O", 12.0),
        ("atomic", "total", 20.0),
    ],
)
def test_renormalization_against_python(unit, component, target):
    data = {"MgO": 1.0, "SiO2": 1.0}
    native, pure = bm.Composition(data, "molar"), ReferenceComposition(data, "molar")
    native.renormalize(unit, component, target)
    pure.renormalize(unit, component, target)
    compare(native, pure)


@pytest.mark.parametrize("unit", ["mass", "weight", "molar"])
def test_addition_and_removal_against_python(unit):
    data = {"MgSiO3": 0.5, "FeSiO3": 0.5}
    native, pure = bm.Composition(data, "molar"), ReferenceComposition(data, "molar")
    addition = {"MgSiO3": 0.1, "Al2O3": 0.05}
    native.add_components(addition, unit)
    pure.add_components(addition, unit)
    compare(native, pure)
    basis = ["MgO", "SiO2", "FeO", "Al2O3", "CaO"]
    native.change_component_set(basis)
    pure.change_component_set(basis)
    compare(native, pure)
    native.remove_null_components()
    pure.remove_null_components()
    compare(native, pure)


def test_arithmetic_against_python():
    a = {"MgO": 1.0, "SiO2": 1.0}
    b = {"MgO": 2.0, "FeO": 3.0}
    native_a, pure_a = bm.Composition(a, "molar"), ReferenceComposition(a, "molar")
    native_b, pure_b = bm.Composition(b, "molar"), ReferenceComposition(b, "molar")
    compare(native_a + native_b, pure_a + pure_b)
    compare(native_a - native_b, pure_a - pure_b)
    compare(native_a * 2.0, pure_a * 2.0)
    compare(native_a / 2.0, pure_a / 2.0)


@pytest.mark.parametrize("unit", ["mass", "weight", "molar", "atomic"])
def test_formatted_output_against_python(unit, capsys):
    data = {"MgO": 1.0, "SiO2": 2.0}
    pure = ReferenceComposition(data, "molar")
    pure.print(unit, significant_figures=3, normalization_amount=100.0)
    expected = capsys.readouterr().out
    native = bm.Composition(data, "molar")
    native.print(unit, significant_figures=3, normalization_amount=100.0)
    assert capsys.readouterr().out == expected
