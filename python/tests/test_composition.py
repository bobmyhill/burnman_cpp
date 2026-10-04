"""Native Composition API, conservation, and error handling."""

import copy
import sys
import subprocess

import numpy as np
import pytest
import burnman_cpp as bm


@pytest.mark.parametrize("unit", ["mass", "weight", "molar"])
@pytest.mark.parametrize("normalize", [False, True])
def test_input_and_output(unit, normalize):
    data = {"SiO2": 3.0, "MgO": 1.0, "FeO": 2.0}
    c = bm.Composition(data, unit_type=unit, normalize=normalize)
    expected = {key: value / (6.0 if normalize else 1.0) for key, value in data.items()}
    assert c.composition(unit) == pytest.approx(expected)
    assert list(c.composition(unit)) == list(data)
    assert c.weight_composition == c.mass_composition
    assert c.get_molar_composition() == c.molar_composition
    assert c.get_atomic_composition() == c.atomic_composition
    assert c.get_component_formulae() == c.component_formulae
    assert c.get_element_list() == c.element_list


def test_conversions_and_snapshot_properties():
    data = {"MgSiO3": 0.5, "FeSiO3": 0.5}
    c = bm.Composition(data, "molar")
    assert c.atomic_composition == pytest.approx(
        {"Mg": 0.5, "Si": 1.0, "O": 3.0, "Fe": 0.5}
    )
    assert c.mass_composition["MgSiO3"] == pytest.approx(
        0.5 * (0.024305 + 0.0280855 + 3 * 0.0159994)
    )
    assert c.element_list == ["Mg", "Si", "O", "Fe"]
    assert c.component_formulae["MgSiO3"] == {"Mg": 1.0, "Si": 1.0, "O": 3.0}
    data["MgSiO3"] = 99.0
    snapshot = c.mass_composition
    snapshot["MgSiO3"] = 99.0
    formula = c.component_formulae
    formula["MgSiO3"]["O"] = 99.0
    assert c.molar_composition["MgSiO3"] == pytest.approx(0.5)
    assert c.atomic_composition["O"] == pytest.approx(3.0)


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
def test_renormalize(unit, component, target):
    c = bm.Composition({"MgO": 1.0, "SiO2": 1.0}, "molar")
    c.renormalize(unit, component, target)
    result = c.composition(unit)
    assert (
        sum(result.values()) if component == "total" else result[component]
    ) == pytest.approx(target)
    assert c.molar_composition["MgO"] == pytest.approx(c.molar_composition["SiO2"])


def test_mixed_addition_signed_arithmetic_and_copies():
    c = bm.Composition({"MgO": 1.0, "SiO2": 1.0}, "molar")
    d = bm.Composition({"MgO": 2.0, "FeO": 3.0}, "molar")
    original = c.mass_composition
    assert (c + d).molar_composition == pytest.approx(
        {"MgO": 3.0, "SiO2": 1.0, "FeO": 3.0}
    )
    assert (c - d).molar_composition == pytest.approx(
        {"MgO": -1.0, "SiO2": 1.0, "FeO": -3.0}
    )
    assert (c * 2 / 2).mass_composition == pytest.approx(original)
    assert (2 * c).mass_composition == pytest.approx(
        {k: 2 * v for k, v in original.items()}
    )
    for clone in [copy.copy(c), copy.deepcopy(c)]:
        clone.add_components(original, "mass")
        assert clone.molar_composition == pytest.approx({"MgO": 2.0, "SiO2": 2.0})
    assert c.mass_composition == original
    alias = c
    c -= d
    assert alias is c
    assert c.molar_composition == pytest.approx({"MgO": -1.0, "SiO2": 1.0, "FeO": -3.0})
    c += d
    c *= 2
    c /= 2
    c.add_components({"MgO": -2.0}, "molar")
    assert c.molar_composition["MgO"] == pytest.approx(-1.0)
    c.remove_null_components()
    assert "FeO" not in c.molar_composition


@pytest.mark.parametrize("scale", [1e-18, 1.0, 1e18])
@pytest.mark.parametrize(
    "basis",
    [
        ["MgO", "SiO2", "FeO", "Al2O3"],
        ["MgO", "Mg2O2", "SiO2", "FeO", "MgSiO3", "FeSiO3"],
    ],
)
def test_component_basis_conserves_atoms(scale, basis):
    c = bm.Composition({"MgSiO3": 0.5 * scale, "FeSiO3": 0.5 * scale}, "molar")
    before = c.atomic_composition
    c.change_component_set(basis)
    after = c.atomic_composition
    for element, value in before.items():
        assert after[element] == pytest.approx(value, rel=1e-11, abs=1e-30)
    assert all(value >= 0 for value in c.molar_composition.values())
    assert list(c.molar_composition) == basis


@pytest.mark.parametrize("seed", range(30))
def test_nnls_rank_and_boundary_cases(seed):
    rng = np.random.default_rng(seed)
    elements = ["Mg", "Fe", "Si", "Al", "Ca", "Na", "O"]
    basis = []
    while len(basis) < 15:
        counts = rng.integers(0, 5, len(elements))
        counts[-1] += 1
        formula = "".join(
            f"{element}{count}" for element, count in zip(elements, counts) if count
        )
        if formula not in basis:
            basis.append(formula)
    coefficients = rng.random(len(basis))
    coefficients[rng.random(len(basis)) < 0.6] = 0
    c = bm.Composition(dict(zip(basis, coefficients)), "molar")
    before = c.atomic_composition
    c.change_component_set(basis[::-1])
    assert c.atomic_composition == pytest.approx(before, rel=1e-10, abs=1e-12)
    assert min(c.molar_composition.values()) >= 0


@pytest.mark.parametrize(
    "basis", [[], ["MgSiO2"], ["SiO2"], ["MgO", "SiO2", "SiO2"], ["MgO", "XxO"]]
)
def test_failed_basis_change_preserves_object(basis):
    c = bm.Composition({"MgO": 1.0, "SiO2": 1.0}, "molar")
    before = c.mass_composition, c.element_list, c.component_formulae
    with pytest.raises(ValueError):
        c.change_component_set(basis)
    assert (c.mass_composition, c.element_list, c.component_formulae) == before


def test_tiny_incompatible_bulk_is_rejected():
    c = bm.Composition({"MgO": 1e-18}, "molar")
    with pytest.raises(ValueError, match="nonnegative representation"):
        c.change_component_set(["SiO2"])


def test_signed_inventory_requires_a_nonnegative_new_representation():
    c = bm.Composition({"MgSiO3": 2.0, "SiO2": -1.0}, "molar")
    c.change_component_set(["MgO", "SiO2"])
    assert c.molar_composition == pytest.approx({"MgO": 2.0, "SiO2": 1.0})
    negative = bm.Composition({"MgO": -1.0}, "molar")
    with pytest.raises(ValueError):
        negative.change_component_set(["MgO"])


def test_fractional_formulae_and_repeated_elements():
    c = bm.Composition({"Mg1/2Fe0.5SiO3": 2.0, "CH3COOH": 1.0}, "molar")
    assert c.atomic_composition == pytest.approx(
        {"Mg": 1.0, "Fe": 1.0, "Si": 2.0, "O": 8.0, "C": 2.0, "H": 4.0}
    )


@pytest.mark.parametrize(
    "formula",
    ["", "2MgO", "XxO", "MgO!", "Mg-1O", "Mg1/0O", "Vc", "Mg(OH)2", "Mg1e9999O"],
)
def test_invalid_formula(formula):
    with pytest.raises(ValueError):
        bm.Composition({formula: 1.0}, "molar")


@pytest.mark.parametrize("unit", ["atomic", "volume", "garbage"])
def test_invalid_input_unit(unit):
    with pytest.raises(ValueError):
        bm.Composition({"MgO": 1.0}, unit)


def test_invalid_operations_leave_object_unchanged():
    c = bm.Composition({"MgO": 1.0, "SiO2": 0.0}, "molar")
    before = c.mass_composition
    operations = [
        lambda: c.renormalize("molar", "SiO2", 1.0),
        lambda: c.renormalize("atomic", "Fe", 1.0),
        lambda: c.renormalize("volume", "total", 1.0),
        lambda: c.add_components({"XxO": 1.0}, "molar"),
        lambda: c.add_components({"MgO": float("nan")}, "mass"),
        lambda: c.remove_null_components(-1.0),
        lambda: c.__imul__(float("inf")),
        lambda: c.__itruediv__(0.0),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
        assert c.mass_composition == before
    with pytest.raises(ValueError):
        bm.Composition({"MgO": 0.0}, "molar", True)
    with pytest.raises(TypeError):
        c + 1.0


def test_empty_and_null_components():
    c = bm.Composition({"MgO": 0.0, "FeO": -1e-6, "SiO2": 1e-15}, "molar")
    c.remove_null_components()
    assert c.molar_composition == pytest.approx({"FeO": -1e-6})
    assert c.element_list == ["Fe", "O"]
    empty = bm.Composition({}, "molar")
    empty.change_component_set(["MgO", "SiO2"])
    assert empty.molar_composition == {"MgO": 0.0, "SiO2": 0.0}
    empty.remove_null_components()
    empty.change_component_set([])
    assert empty.atomic_composition == {}


def test_pretty_print_does_not_mutate(capsys):
    c = bm.Composition({"SiO2": 1.0, "MgO": 1.0}, "molar")
    before = c.mass_composition
    c.print("molar", significant_figures=2, normalization_amount=100.0)
    expected = "Molar composition\nMgO: 50.00\nSiO2: 50.00\n"
    assert capsys.readouterr().out == expected
    assert c.format("molar", 2, normalization_amount=100.0) == expected
    assert c.mass_composition == before
    assert "Composition (mass)" in repr(c)


def test_read_composition_table(tmp_path):
    table = tmp_path / "compositions.txt"
    table.write_text(
        "# Oxide table\n\n  # another comment\nMgO SiO2 Comment\n1 1 basalt sample\n3 1\n"
    )
    compositions, comments = bm.file_to_composition_list(str(table), "molar", True)
    assert [c.molar_composition for c in compositions] == [
        pytest.approx({"MgO": 0.5, "SiO2": 0.5}),
        pytest.approx({"MgO": 0.75, "SiO2": 0.25}),
    ]
    assert comments == [["basalt", "sample"], []]
    table.write_text("MgO SiO2 Comment\n1 nope\n")
    with pytest.raises(ValueError, match="line 2"):
        bm.file_to_composition_list(str(table), "molar", False)


@pytest.mark.parametrize("row", ["1 2garbage", "1 nan", "1 inf", "1 1e9999", "1"])
def test_read_composition_table_rejects_bad_numbers(tmp_path, row):
    table = tmp_path / "bad.txt"
    table.write_text(f"MgO SiO2 Comment\n{row}\n")
    with pytest.raises(ValueError, match="line 2"):
        bm.file_to_composition_list(str(table), "mass", False)


def test_no_python_burnman_or_scipy_dependency():
    script = """
import builtins
original = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in ('burnman', 'scipy'):
        raise AssertionError('Unexpected Python dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked_import
from burnman_cpp import Composition
c = Composition({'MgSiO3': 1.0}, 'molar')
c.change_component_set(['MgO', 'SiO2'])
assert abs(c.molar_composition['MgO'] - 1.0) < 1e-12
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
