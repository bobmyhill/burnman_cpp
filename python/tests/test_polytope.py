from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import JH_2015 as JH, SLB_2011 as SLB, HGP_2018_ds633 as HGP
from conftest import oxide_params


def assemblage(phases):
    return bm.Assemblage(phases, np.full(len(phases), 1.0 / len(phases)))


def test_cdd_vertices_equalities_empty_and_unbounded():
    # A rational line segment with redundant equalities, x + y = 1/3.
    eq = np.array([[-1.0 / 3.0, 1.0, 1.0], [-2.0 / 3.0, 2.0, 2.0]])
    inequalities = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    p = bm.MaterialPolytope(eq, inequalities)
    assert p.is_bounded and not p.is_empty
    assert p.vertices.shape == (2, 2)
    np.testing.assert_allclose(np.sort(p.vertices, axis=0), [[0, 0], [1 / 3, 1 / 3]])
    np.testing.assert_allclose(
        eq[:, 1:] @ p.vertices.T, np.repeat(-eq[:, :1], 2, axis=1), atol=1e-14
    )
    empty = bm.MaterialPolytope([[-1.0, 1.0], [-2.0, 1.0]], [[0.0, 1.0]])
    assert empty.is_empty and empty.vertices.shape == (0, 1)
    unbounded = bm.MaterialPolytope(np.empty((0, 2)), [[0.0, 1.0]])
    assert not unbounded.is_bounded and not unbounded.is_empty
    assert unbounded.rays.shape == (1, 1)
    copy = p.vertices
    copy[:] = -100
    assert np.all(p.vertices >= 0)


def test_threaded_cdd_calls():
    def vertices(_):
        p = bm.MaterialPolytope([[-1.0, 1.0, 1.0]], [[0, 1, 0], [0, 0, 1]])
        return p.vertices

    with ThreadPoolExecutor(max_workers=4) as executor:
        for v in executor.map(vertices, range(20)):
            np.testing.assert_allclose(np.sort(v, axis=0), [[0, 0], [1, 1]])


def test_remove_impossible_phases_and_collapse_single_endmember():
    original = assemblage(
        [SLB.mg_fe_bridgmanite(), SLB.mg_fe_olivine(), SLB.ferropericlase()]
    )
    reduced = bm.simplify_composite_with_composition(original, dict(Mg=1, Si=1, O=3))
    assert len(reduced.phases) == 1
    assert not isinstance(reduced.phases[0], bm.Solution)
    assert reduced.phases[0].formula == dict(Mg=1, Si=1, O=3)
    assert len(original.phases) == 3
    assert original.phases[0].n_endmembers == 3


def test_multiple_solutions_reduce_at_joint_bulk_composition():
    original = assemblage([SLB.mg_fe_olivine(), SLB.orthopyroxene()])
    composition = dict(Mg=2, Si=1.5, O=5)
    poly = bm.composite_polytope_at_constrained_composition(original, composition)
    assert poly.is_bounded and not poly.is_empty
    amounts = poly.endmember_occupancies
    bulk = np.array([composition.get(e, 0) for e in original.elements])
    np.testing.assert_allclose(
        amounts @ original.stoichiometric_matrix, np.tile(bulk, (len(amounts), 1))
    )
    reduced = bm.simplify_composite_with_composition(original, composition)
    assert len(reduced.phases) == 2
    assert all(not isinstance(p, bm.Solution) for p in reduced.phases)
    assert all("Fe" not in p.formula for p in reduced.phases)


def test_interior_fixed_composition_retains_binary_solution(solution):
    # Fixing a bulk ratio does not make a solid solution a pure endmember.
    original = assemblage([solution])
    reduced = bm.simplify_composite_with_composition(
        original, dict(Mg=0.7, Fe=0.3, O=1)
    )
    child = reduced.phases[0]
    assert isinstance(child, bm.Solution) and child.n_endmembers == 2
    child.set_state(10e9, 1800)
    solution.set_state(1e9, 1200)
    expected = solution.molar_gibbs
    child.set_state(25e9, 2200)
    solution.reset_cache()
    assert solution.molar_gibbs == expected


def test_signed_majorite_basis_preserves_initial_composition():
    garnet = SLB.garnet()
    garnet.set_composition([-0.1, 0.1, 0.0, 1.0, 0.0])
    original = assemblage([SLB.mg_fe_olivine(), garnet])
    reduced = bm.simplify_composite_with_composition(
        original, dict(Fe=3, Mg=1, Si=3.9, O=11.8)
    )
    child = reduced.phases[1]
    assert child.n_endmembers == 2
    assert child.basis.min() < 0
    np.testing.assert_allclose(
        child.basis.T @ child.molar_fractions, garnet.molar_fractions, atol=1e-12
    )
    garnet.set_state(5e9, 1400)
    child.set_state(5e9, 1400)
    assert child.molar_gibbs == pytest.approx(garnet.molar_gibbs, rel=1e-12)


def test_repeated_simplification_and_state_preservation():
    original = assemblage([SLB.garnet()])
    original.n_moles = 2.5
    original.set_state(1e9, 1200)
    first = bm.simplify_composite_with_composition(
        original, dict(Mg=1.5, Ca=1.5, Al=2, Si=3, O=12)
    )
    second = bm.simplify_composite_with_composition(first, dict(Mg=3, Al=2, Si=3, O=12))
    assert second.n_moles == original.n_moles
    assert second.pressure == original.pressure
    assert second.temperature == original.temperature
    assert second.phases[0].formula == dict(Mg=3, Al=2, Si=3, O=12)
    assert np.isfinite(second.phases[0].molar_gibbs)


@pytest.mark.parametrize(
    "factory,composition,n",
    [
        (JH.orthopyroxene, dict(Mg=1, Fe=1, Si=2, O=6), 3),
        (SLB.garnet, dict(Mg=1.5, Ca=1.5, Al=2, Si=3, O=12), 2),
        (SLB.mg_fe_bridgmanite, dict(Mg=0.9, Fe=0.1, Si=1, O=3), 2),
    ],
)
def test_general_simplification_matches_example_thermodynamics(factory, composition, n):
    original = factory()
    reduced = bm.simplify_composite_with_composition(
        assemblage([original]), composition
    ).phases[0]
    assert reduced.n_endmembers == n
    assert reduced.basis.shape == (n, original.n_endmembers)
    for fractions in [np.full(n, 1 / n), np.arange(1, n + 1) / (n * (n + 1) / 2)]:
        reduced.set_composition(fractions)
        original.set_composition(reduced.basis.T @ fractions)
        for pressure, temperature in [(1e5, 800), (5e9, 1500)]:
            reduced.set_state(pressure, temperature)
            original.set_state(pressure, temperature)
            for prop in ["molar_gibbs", "molar_entropy", "molar_volume"]:
                assert getattr(reduced, prop) == pytest.approx(
                    getattr(original, prop), rel=2e-12, abs=1e-10
                )


def test_complete_melt_simplifies_for_chromium_free_bulk():
    original = HGP.silicate_melt()
    chromium = original.stoichiometric_matrix[:, original.elements.index("Cr")] != 0.0
    fractions = np.arange(1.0, original.n_endmembers + 1)
    fractions[chromium] = 0.0
    fractions /= fractions.sum()
    original.set_composition(fractions)
    reduced = bm.simplify_composite_with_composition(
        assemblage([original]), original.formula
    ).phases[0]
    assert original.n_endmembers == 12
    assert reduced.n_endmembers == 11
    np.testing.assert_allclose(reduced.basis[:, chromium], 0.0, atol=1e-12)
    for fractions in [np.full(11, 1 / 11), np.arange(1.0, 12) / 66]:
        reduced.set_composition(fractions)
        original.set_composition(reduced.basis.T @ fractions)
        assert original.formula.get("Cr", 0.0) == 0.0
        assert reduced.formula.get("Cr", 0.0) == 0.0
        for pressure, temperature in [(1e5, 700), (3e9, 1200)]:
            original.set_state(pressure, temperature)
            reduced.set_state(pressure, temperature)
            for prop in ["molar_gibbs", "molar_entropy", "molar_volume"]:
                assert getattr(reduced, prop) == pytest.approx(
                    getattr(original, prop), rel=2e-12, abs=1e-10
                )


def test_melt_repeated_simplification_preserves_empty_sites():
    original = HGP.silicate_melt()
    first = bm.simplify_composite_with_composition(
        assemblage([original]), dict(Mg=2, Si=2, O=7, H=2)
    )
    second = bm.simplify_composite_with_composition(first, dict(Si=2, O=5, H=2))
    assert first.phases[0].n_endmembers == 3
    child = second.phases[0]
    assert child.n_endmembers == 2
    original.set_composition(child.basis.T @ child.molar_fractions)
    for phase in (original, child):
        phase.set_state(3e9, 1200)
    assert child.molar_gibbs == pytest.approx(original.molar_gibbs, rel=2e-12)


@pytest.mark.parametrize(
    "kind",
    [bm.IdealSolution, bm.SymmetricRegularSolution, bm.AsymmetricRegularSolution],
)
def test_basis_transformation_preserves_thermodynamics_and_derivatives(kind):
    members = [
        (bm.Mineral(oxide_params(e, shift)), f"[{e}]O")
        for e, shift in [("Mg", 0), ("Fe", 1000), ("Ca", 2500)]
    ]
    kwargs = {}
    if kind != bm.IdealSolution:
        kwargs = dict(
            energy_interaction=[[3000, 8000], [5000]],
            entropy_interaction=[[1, 2], [3]],
            volume_interaction=[[1e-7, 2e-7], [3e-7]],
        )
    if kind == bm.AsymmetricRegularSolution:
        kwargs["alphas"] = [1, 2, 3]
    original = bm.Solution(kind(members, **kwargs), [0.3, 0.3, 0.4])
    basis = np.array([[0.6, 0.4, 0.0], [0.0, 0.2, 0.8]])
    reduced = bm.transform_solution_to_new_basis(
        original, basis, solution_name="test reduction"
    )
    assert reduced.name == "test reduction"
    for p in [[0.15, 0.85], [0.7, 0.3]]:
        reduced.set_composition(p)
        original.set_composition(basis.T @ p)
        for pressure, temperature in [(1e5, 600), (5e9, 1400)]:
            original.set_state(pressure, temperature)
            reduced.set_state(pressure, temperature)
            for prop in [
                "molar_gibbs",
                "molar_entropy",
                "molar_volume",
                "molar_heat_capacity_p",
            ]:
                assert getattr(reduced, prop) == pytest.approx(
                    getattr(original, prop), rel=5e-12, abs=1e-9
                )
            np.testing.assert_allclose(
                reduced.partial_gibbs,
                basis @ original.partial_gibbs,
                rtol=1e-11,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                reduced.gibbs_hessian,
                basis @ original.gibbs_hessian @ basis.T,
                rtol=1e-11,
                atol=1e-7,
            )


def test_transform_single_mixed_endmember_and_existing_mixed_standard_state():
    mg, fe = bm.Mineral(oxide_params("Mg")), bm.Mineral(oxide_params("Fe", 1000))
    mixed = bm.CombinedMineral([mg, fe], [0.5, 0.5])
    # Original model already has a nonzero standard-state configurational entropy.
    model = bm.IdealSolution([(mg, "[Mg]O"), (mixed, "[Mg1/2Fe1/2]O")])
    original = bm.Solution(model, [0.3, 0.7])
    basis = np.array([[0.3, 0.7]])
    reduced = bm.transform_solution_to_new_basis(original, basis)
    assert not isinstance(reduced, bm.Solution)
    original.set_state(1e9, 1000)
    reduced.set_state(1e9, 1000)
    assert reduced.molar_gibbs == pytest.approx(original.molar_gibbs, rel=1e-12)
    assert reduced.molar_entropy == pytest.approx(original.molar_entropy, rel=1e-12)


def test_signed_physical_vertices_and_invalid_inputs():
    opx = JH.mg_fe_orthopyroxene()
    poly = bm.solution_polytope_from_endmember_occupancies(opx.endmember_occupancies)
    assert poly.vertices.min() < 0
    assert poly.endmember_occupancies.min() >= -1e-12
    assert poly.vertices.shape[0] == 4
    original = assemblage([SLB.mg_fe_olivine()])
    for bulk in [
        dict(Mg=1, Si=1, O=3),
        dict(Ca=1),
        dict(Mg=-1),
        dict(Mg=float("nan")),
        {},
    ]:
        with pytest.raises(ValueError):
            bm.simplify_composite_with_composition(original, bulk)
    for basis in [[[0.3, 0.3, 0.3]], [[1, 0]], [[2, -1, 0]], [[1, 0, 0], [1, 0, 0]]]:
        with pytest.raises(ValueError):
            bm.transform_solution_to_new_basis(opx, basis)
    with pytest.raises(ValueError):
        bm.MaterialPolytope([[0, float("nan")]], [[0, 1]])
    with pytest.raises(ValueError, match="bounded"):
        bm.solution_polytope_from_endmember_occupancies([[1, 0], [1, 0]])
