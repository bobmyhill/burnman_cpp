"""Optional comparisons against the original Python mineral models."""

import numpy as np
import pytest

burnman = pytest.importorskip("burnman")
from burnman.minerals import HP_2011_ds62 as HP, SLB_2011 as SLB, JH_2015 as JH
from burnman.tools.polytope import simplify_composite_with_composition

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman
from burnman_cpp.minerals import SLB_2011 as native_SLB, JH_2015 as native_JH
from burnman_cpp.minerals import mp50NCKFMASHTO as native_MP
from burnman.minerals import mp50NCKFMASHTO as reference_MP


@pytest.mark.parametrize(
    "name",
    [
        "g",
        "pl4tr",
        "k4tr",
        "plc",
        "ksp",
        "ep",
        "ma",
        "mu",
        "bi",
        "opx",
        "sa",
        "cd",
        "st",
        "chl",
        "ctd",
        "sp",
        "ilmm",
        "ilm",
        "mt1",
    ],
)
def test_native_metapelite_models_match_published_reference(name):
    reference = getattr(reference_MP, name)()
    native = getattr(native_MP, name)()
    fractions = np.arange(1.0, native.n_endmembers + 1)
    fractions /= fractions.sum()
    reference.set_composition(fractions)
    native.set_composition(fractions)
    assert native.formula == pytest.approx(reference.formula)
    for pressure, temperature in [(2.0e8, 673.15), (1.7e9, 1123.15)]:
        reference.set_state(pressure, temperature)
        native.set_state(pressure, temperature)
        for property_name in [
            "molar_gibbs",
            "molar_entropy",
            "molar_volume",
            "molar_heat_capacity_p",
            "partial_gibbs",
            "gibbs_hessian",
        ]:
            np.testing.assert_allclose(
                getattr(native, property_name),
                getattr(reference, property_name),
                rtol=2.0e-7,
                atol=2.0e-5,
                err_msg=f"{name}: {property_name}",
            )


@pytest.mark.parametrize(
    "source", [HP.sill, HP.mst, HP.fst, SLB.ferropericlase, JH.orthopyroxene]
)
def test_adapter_thermodynamic_properties(source):
    reference = source()
    if isinstance(reference, burnman.Solution):
        reference.set_composition(
            np.full(reference.n_endmembers, 1.0 / reference.n_endmembers)
        )
    native = from_burnman(reference)
    reference.set_state(3.0e9, 1200.0)
    native.set_state(3.0e9, 1200.0)
    # Test the free energy and derivatives used by equilibration. The native
    # Bragg-Williams pressure finite differences amplify rounding in the
    # second derivative, so its bulk modulus is not an exact match.
    for property_name in (
        "molar_gibbs",
        "molar_volume",
        "molar_entropy",
        "molar_heat_capacity_p",
        "thermal_expansivity",
    ):
        np.testing.assert_allclose(
            getattr(native, property_name),
            getattr(reference, property_name),
            rtol=1.0e-6,
            atol=1.0e-10,
            err_msg=property_name,
        )


@pytest.mark.parametrize("method", ["hp_tmt", "mgd3", "slb3"])
@pytest.mark.parametrize("pressure,temperature", [(1.0e5, 298.15), (3.0e9, 1200.0)])
def test_fractional_formula_units_match_python(method, pressure, temperature):
    reference = HP.mst() if method == "hp_tmt" else SLB.forsterite()
    params = dict(reference.params)
    params["equation_of_state"] = method
    # Half a formula unit exercises every thermal EOS with a fractional n.
    params["n"] *= 0.5
    params["formula"] = {
        element: amount * 0.5 for element, amount in params["formula"].items()
    }
    for key in ("molar_mass", "V_0", "H_0", "S_0", "F_0", "E_0"):
        if key in params:
            params[key] *= 0.5
    if "Cp" in params:
        params["Cp"] = (np.asarray(params["Cp"]) * 0.5).tolist()
    reference = burnman.Mineral(params)
    native = from_burnman(reference)
    assert native.params.napfu == params["n"]
    assert native.formula == params["formula"]
    reference.set_state(pressure, temperature)
    native.set_state(pressure, temperature)
    for prop in (
        "molar_gibbs",
        "molar_entropy",
        "molar_volume",
        "molar_heat_capacity_p",
        "molar_heat_capacity_v",
        "thermal_expansivity",
    ):
        assert getattr(native, prop) == pytest.approx(
            getattr(reference, prop), rel=1.0e-6, abs=1.0e-12
        ), prop


def test_transformed_ordering_solution_signed_coordinates():
    composition = dict(Mg=1.0, Fe=1.0, Si=2.0, O=6.0)
    reference = simplify_composite_with_composition(
        burnman.Composite([JH.orthopyroxene()]), composition
    ).phases[0]
    reference.set_composition([0.55, -0.1, 0.55])
    native = from_burnman(reference)
    native.set_state(1.0e5, 800.0)
    reference.set_state(1.0e5, 800.0)
    assert native.molar_gibbs == pytest.approx(reference.molar_gibbs, rel=1.0e-8)
    a = bm.Assemblage([native], [1.0])
    a.set_state(1.0e5, 800.0)
    x = bm.get_parameter_vector(a)
    bm.set_composition_and_state_from_parameters(a, x)
    np.testing.assert_allclose(native.molar_fractions, [0.55, -0.1, 0.55])
    with pytest.raises(ValueError, match="occupancies"):
        native.set_composition([1.1, -1.2, 1.1])


@pytest.mark.parametrize(
    "source,native_source",
    [
        (SLB.garnet, native_SLB.garnet),
        (JH.orthopyroxene, native_JH.orthopyroxene),
    ],
)
def test_native_polytope_vertices_match_reference(source, native_source):
    from burnman.tools.polytope import solution_polytope_from_endmember_occupancies

    reference = source()
    native = native_source()
    occupancies = reference.solution_model.endmember_occupancies
    expected = solution_polytope_from_endmember_occupancies(occupancies)
    actual = bm.solution_polytope_from_endmember_occupancies(
        native.endmember_occupancies
    )
    expected_vertices = np.asarray(
        expected.endmembers_as_independent_endmember_amounts, dtype=float
    )
    assert actual.vertices.shape == expected_vertices.shape
    distances = np.linalg.norm(
        actual.vertices[:, None, :] - expected_vertices[None, :, :], axis=2
    )
    assert np.max(np.min(distances, axis=0)) < 1.0e-10
    assert np.max(np.min(distances, axis=1)) < 1.0e-10
