"""Structural equilibration, thermoelastic derivatives and Python parity."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from test_anisotropic_mineral import model_parameters, python_reference
from test_anisotropic_solution import endmember_data, excess_psi

Q = np.array([[1.0, -1.0, 0.0]])
X = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])


def make_unrelaxed(system="triclinic", reference=None, fractions=None):
    # Two isochemical Mg states with different energies, volumes and shapes.
    data = endmember_data(system)
    scalar, cell, c = model_parameters(system, thermal=True)
    params = scalar.params
    params.F_0 += 1500.0
    params.V_0 *= 1.025
    scalar = bm.Mineral(params)
    cell[:3] *= np.array([1.03, 1.0, 1.025 / 1.03])
    c[:3, :3, 1, 0] += np.diag([0.03, -0.02, -0.01])
    data.insert(1, (scalar, cell, c, "[Mg]2SiO4"))
    if reference:
        minerals = pytest.importorskip("burnman.minerals.SLB_2011")
        models = pytest.importorskip("burnman.classes.solutionmodel")
        module = pytest.importorskip("burnman.classes.anisotropicsolution")
        scalars = [minerals.forsterite(), minerals.forsterite(), minerals.fayalite()]
        scalars[1].params["F_0"] += 1500.0
        scalars[1].params["V_0"] *= 1.025
        endmembers = [
            (reference.AnisotropicMineral(m, cell, c, [0, 1, 2]), formula)
            for m, (_, cell, c, formula) in zip(scalars, data)
        ]
    else:
        models, module = bm, bm
        endmembers = [
            (bm.AnisotropicMineral(m, cell, c, [0, 1, 2]), formula)
            for m, cell, c, formula in data
        ]
    model = models.SymmetricRegularSolution(
        endmembers,
        [[-100000.0, 10000.0], [10000.0]],
        volume_interaction=[[0.0, 1.0e-6], [1.0e-6]],
    )
    solution = module.AnisotropicSolution(
        "structural olivine", model, excess_psi, {"coefficient": 0.025}
    )
    if fractions is not None:
        solution.set_composition(fractions)
    return solution


def make_relaxed(system="triclinic"):
    solution = bm.RelaxedAnisotropicSolution(make_unrelaxed(system), Q, X)
    solution.set_composition([0.8, 0.2])
    solution.set_state(20.0e9, 1000.0)
    return solution


@pytest.mark.parametrize("system", ["orthorhombic", "monoclinic", "triclinic"])
def test_reference_hessians_and_responses(system, python_reference):
    module = pytest.importorskip("burnman.classes.anisotropicsolution")
    original = module.RelaxedAnisotropicSolution(
        make_unrelaxed(system, python_reference), Q, X
    )
    original.set_composition([0.8, 0.2], relaxed=False)
    original.set_state(20.0e9, 1000.0)
    native = make_relaxed(system)
    # Separate the Nelder-Mead stopping tolerance from the tensor formulas.
    assert_allclose(native.molar_fractions, original.molar_fractions, atol=3.0e-5)
    q = np.linalg.lstsq(Q.T, original.molar_fractions - X.T @ [0.8, 0.2], rcond=None)[0]
    native.set_composition([0.8, 0.2], q, relaxed=False)
    native.set_state(20.0e9, 1000.0, relaxed=False)
    for name in [
        "_depsdq_fixed_VT",
        "_depsdT_fixed_volume",
        "_dbarepsdeps",
        "_dSdq",
        "_dPdq",
        "_d2Fdqdq_fixed_VT",
        "_d2Fdqdq_fixed_epsT",
        "_d2Fdqdq_fixed_epsT_pinv",
        "_d2Fdqdz",
        "dqdz_relaxed",
        "_d2Fdzdz_Q",
        "_d2Fdzdz",
        "isothermal_stiffness_tensor",
        "isentropic_stiffness_tensor",
        "full_isothermal_compliance_tensor",
        "thermal_stress_tensor",
        "thermal_expansivity_tensor",
        "grueneisen_tensor",
        "molar_isometric_heat_capacity",
        "molar_heat_capacity_p",
        "molar_heat_capacity_v",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "cell_vectors",
    ]:
        expected = getattr(original, name)
        scale = max(np.max(np.abs(expected)), 1.0e-30)
        assert_allclose(
            np.asarray(getattr(native, name)) / scale,
            np.asarray(expected) / scale,
            rtol=1.0e-7,
            atol=1.0e-9,
            err_msg=name,
        )


@pytest.mark.parametrize("system", ["orthorhombic", "triclinic"])
def test_equilibrium_and_hydrostatic_derivatives(system):
    s = make_relaxed(system)
    assert_allclose(Q @ s.partial_gibbs, 0.0, atol=1.0e-6)
    assert_allclose(s.molar_fractions, s.unrelaxed.molar_fractions)
    assert_allclose(s.unrelaxed_vectors, [0.8, 0.2], atol=1e-15)
    assert np.all(s.molar_fractions > 0.0)
    assert_allclose(np.linalg.det(s.cell_vectors), s.molar_volume, rtol=1.0e-12)
    assert_allclose(
        s.isothermal_stiffness_tensor @ s.isothermal_compliance_tensor,
        np.eye(6),
        atol=1.0e-13,
    )
    frozen_ct = s.unrelaxed.isothermal_stiffness_tensor
    assert np.linalg.eigvalsh(frozen_ct - s.isothermal_stiffness_tensor).min() > -1.0e-3
    assert s.molar_isometric_heat_capacity >= s.unrelaxed.molar_isometric_heat_capacity
    assert_allclose(s._d2Fdzdz, s._d2Fdzdz_Q + s._d2Fdqdz.T @ s.dqdz_relaxed)
    volume, entropy = s.molar_volume, s.molar_entropy
    inverse_cell = np.linalg.inv(s.cell_vectors.T)
    rotation = s.rotation_matrix
    kt, alpha, cp = (
        s.isothermal_bulk_modulus_reuss,
        s.thermal_expansivity,
        s.molar_heat_capacity_p,
    )
    beta_tensor, alpha_tensor = (
        s.isothermal_compressibility_tensor,
        s.thermal_expansivity_tensor,
    )
    dp, dt = 1.0e5, 0.01
    values = []
    for p, t in [
        (20e9 - dp, 1000.0),
        (20e9 + dp, 1000.0),
        (20e9, 1000.0 - dt),
        (20e9, 1000.0 + dt),
    ]:
        s.set_state(p, t)
        # Hold the crystal frame fixed while differentiating cell deformation.
        values.append(
            (s.molar_volume, s.molar_entropy, rotation @ s.unrotated_cell_vectors.T)
        )
    assert_allclose(
        -(values[1][0] - values[0][0]) / (2 * dp * volume), 1 / kt, rtol=2e-7
    )
    assert_allclose((values[3][0] - values[2][0]) / (2 * dt * volume), alpha, rtol=2e-7)
    assert_allclose(1000.0 * (values[3][1] - values[2][1]) / (2 * dt), cp, rtol=2e-7)
    for a, b, step, expected, sign in [
        (0, 1, dp, beta_tensor, -1.0),
        (2, 3, dt, alpha_tensor, 1.0),
    ]:
        L = (values[b][2] - values[a][2]) / (2 * step) @ inverse_cell
        assert_allclose(
            sign * 0.5 * (L + L.T),
            expected,
            rtol=3e-6,
            atol=np.max(np.abs(expected)) * 1e-7,
        )
    s.set_state_with_volume(volume, 1000.0)
    assert_allclose(s.pressure, 20e9, rtol=1e-9)
    assert_allclose(s.molar_volume, volume, rtol=1e-12)
    assert_allclose(s.molar_entropy, entropy, rtol=1e-11)
    assert_allclose(Q @ s.partial_gibbs, 0.0, atol=1e-6)


def test_initialization_freezing_copy_and_evaluation():
    source = make_unrelaxed(fractions=[0.3, 0.5, 0.2])
    source.set_state(20e9, 1000.0)
    s = bm.RelaxedAnisotropicSolution(source, Q, X)
    assert s.unrelaxed is source
    assert_allclose(Q @ s.partial_gibbs, 0.0, atol=1e-6)
    s.set_composition([0.8, 0.2], [0.05], relaxed=False)
    s.set_state(10e9, 800.0, relaxed=False)
    assert_allclose(s.molar_fractions, [0.45, 0.35, 0.2])
    assert s.unrelaxed.pressure == s.pressure
    assert_allclose(s.unrelaxed.molar_fractions, s.molar_fractions)
    copy = s.copy()
    assert_allclose(copy.isothermal_stiffness_tensor, s.isothermal_stiffness_tensor)
    assert copy.unrelaxed is not source
    copy.set_composition([0.6, 0.4])
    assert_allclose(s.molar_fractions, [0.45, 0.35, 0.2])
    fractions = s.molar_fractions.copy()
    stiffness = s.isothermal_stiffness_tensor
    values = s.evaluate(
        ["molar_volume", "cell_vectors"],
        [1e9, 20e9],
        [300.0, 1000.0],
        [[0.6, 0.4], [0.8, 0.2]],
    )
    assert values[0].shape == (2,)
    assert values[1].shape == (2, 3, 3)
    assert_allclose(s.molar_fractions, fractions)
    assert s.pressure == 10e9
    s.reset()
    assert_allclose(s.isothermal_stiffness_tensor, stiffness)
    s.q_initial[:] = 100.0
    assert_allclose(s.q_initial, [0.05])


def test_zero_modes_and_validation():
    source = make_unrelaxed()
    s = bm.RelaxedAnisotropicSolution(source, np.empty((0, 3)), np.eye(3))
    with pytest.raises(RuntimeError, match="set_composition"):
        s.set_state(1e9, 300.0)
    s.set_composition([0.3, 0.5, 0.2])
    s.set_state(20e9, 1000.0)
    assert s.dqdz_relaxed.shape == (0, 7)
    for name in [
        "isothermal_stiffness_tensor",
        "thermal_expansivity_tensor",
        "molar_isometric_heat_capacity",
        "molar_heat_capacity_p",
    ]:
        assert_allclose(getattr(s, name), getattr(source, name), rtol=1e-12, atol=1e-12)
    for q, x in [
        ([[1.0, 0.0, -1.0]], X),
        ([[1.0, 1.0, 0.0]], X),
        (Q, [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
    ]:
        with pytest.raises(ValueError):
            bm.RelaxedAnisotropicSolution(make_unrelaxed(), q, x)
    s = make_relaxed()
    for x in [[0.2, 0.2], [-0.2, 1.2], [np.nan, 0.0], [1.0]]:
        with pytest.raises(ValueError):
            s.set_composition(x)
    for q in [[1.0, 2.0], [np.nan]]:
        with pytest.raises(ValueError):
            s.set_composition([0.8, 0.2], q)
    with pytest.raises(ValueError):
        s.set_state_with_volume(-1.0, 1000.0)
    s.set_state(1e9, 0.0)
    with pytest.raises(ValueError, match="positive temperature"):
        _ = s.isothermal_stiffness_tensor


def test_multiple_modes_and_basis_invariance():
    a, b, _ = make_unrelaxed().endmembers
    model = bm.SymmetricRegularSolution([a, b, a], [[-100000.0, -80000.0], [-90000.0]])
    source = bm.AnisotropicSolution(
        "two structural modes", model, excess_psi, {"coefficient": 0.025}
    )
    q = np.array([[1.0, -1.0, 0.0], [0.5, 0.5, -1.0]])
    transform = np.array([[2.0, 0.5], [-1.0, 1.5]])
    solutions = [
        bm.RelaxedAnisotropicSolution(source.copy(), basis, [[1 / 3, 1 / 3, 1 / 3]])
        for basis in [q, transform @ q]
    ]
    for s in solutions:
        s.set_composition([1.0])
        s.set_state(20e9, 1000.0)
        assert s._depsdq_fixed_VT.shape == (3, 3, 2)
        assert s.dqdz_relaxed.shape == (2, 7)
        assert_allclose(q @ s.partial_gibbs, 0.0, atol=1e-6)
    for name in [
        "molar_fractions",
        "isothermal_stiffness_tensor",
        "thermal_expansivity_tensor",
        "molar_heat_capacity_p",
    ]:
        assert_allclose(
            getattr(solutions[0], name), getattr(solutions[1], name), rtol=1e-10
        )


def test_site_constraints_in_an_ordering_mode():
    endmembers = []
    for i, formula in enumerate(["[Mg][Fe]SiO4", "[Fe][Mg]SiO4"]):
        scalar, cell, c = model_parameters()
        params = scalar.params
        params.formula = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        params.molar_mass += 0.03154
        params.F_0 += 1500.0 * i
        endmembers.append((bm.AnisotropicMineral(bm.Mineral(params), cell, c), formula))
    model = bm.IdealSolution(endmembers)
    source = bm.AnisotropicSolution("ordering", model, excess_psi, {})
    s = bm.RelaxedAnisotropicSolution(source, [[1.0, -1.0]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="negative site"):
        s.set_composition([1.0], [0.6])
    s.set_composition([1.0], [0.49])
    s.set_state(20e9, 1000.0)
    assert np.all(s.molar_fractions > 0.0)
    assert_allclose(np.array([1.0, -1.0]) @ s.partial_gibbs, 0.0, atol=1e-6)
