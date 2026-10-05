"""Regressions for the native failures, compared with pure Python BurnMan.

Reference: geodynamics/burnman at 69e700647ae7efeed91dfbce147c036bce619516.
The shared test configuration verifies the imported checkout and commit.
Ordinary reference results are checked separately from tightly solved volumes;
the latter change only scipy's root tolerance, without changing the EOS.
"""

import ast
from pathlib import Path
import re

import numpy as np
import pytest

burnman = pytest.importorskip("burnman")
import scipy.optimize
from burnman.classes import solutionmodel as models
from burnman.eos import property_modifiers
from burnman.tools import equilibration

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def fixture_mineral(name):
    """Use exactly the scalar parameters and formulas in the native fixtures."""
    source = (
        Path(__file__).resolve().parents[2] / "tests/include/solution_fixtures.hpp"
    ).read_text()
    params = {}
    for key, value in re.findall(rf"{name}\.params\.(\w+)\s*=\s*(.*?);", source, re.S):
        if key == "equation_of_state":
            params[key] = "slb3"
        elif key == "formula":
            params[key] = {
                k: float(v) for k, v in re.findall(r'\{"(\w+)",\s*([\d.]+)\}', value)
            }
        else:
            params[{"napfu": "n", "debye_0": "Debye_0"}.get(key, key)] = (
                ast.literal_eval(value.strip())
            )
    assert params, name
    return burnman.Mineral(params)


def bridgmanite():
    endmembers = [
        (fixture_mineral("mg_si_perovskite"), "[Mg][Si]O3"),
        (fixture_mineral("fe_si_perovskite"), "[Fe][Si]O3"),
        (fixture_mineral("al_al_perovskite"), "[Al][Al]O3"),
    ]
    return burnman.Solution(
        name="Bridgmanite",
        solution_model=models.IdealSolution(endmembers),
        molar_fractions=[0.88, 0.07, 0.05],
    )


def pyrolite():
    wuestite = fixture_mineral("wuestite")
    wuestite.property_modifiers = [
        ["linear", {"delta_V": 1.0, "delta_S": 2.0, "delta_E": 3.0}]
    ]
    fp = burnman.Solution(
        name="Ferro-periclase",
        solution_model=models.SymmetricRegularSolution(
            [(fixture_mineral("periclase"), "[Mg]O"), (wuestite, "[Fe]O")],
            energy_interaction=[[13.0e3]],
        ),
        molar_fractions=[0.9, 0.1],
    )
    return burnman.Composite(
        [bridgmanite(), fp, fixture_mineral("ca_perovskite")], [0.7, 0.2, 0.1]
    )


def mgd(modified=False):
    mineral = burnman.Mineral(
        dict(
            equation_of_state="mgd3",
            V_0=11.24e-6,
            K_0=161e9,
            Kprime_0=3.8,
            G_0=131e9,
            Gprime_0=2.1,
            molar_mass=0.0403,
            n=2,
            Debye_0=773.0,
            grueneisen_0=1.5,
            q_0=1.5,
        )
    )
    if modified:
        mineral.property_modifiers = [
            [
                "magnetic_chs",
                dict(
                    structural_parameter=0.4,
                    curie_temperature=[800.0, 1e-8],
                    magnetic_moment=[2.2, 1e-10],
                ),
            ]
        ]
    return mineral


PROPERTIES = [
    "molar_internal_energy",
    "molar_gibbs",
    "molar_helmholtz",
    "molar_mass",
    "molar_volume",
    "density",
    "molar_entropy",
    "molar_enthalpy",
    "isothermal_bulk_modulus_reuss",
    "isentropic_bulk_modulus_reuss",
    "isothermal_compressibility_reuss",
    "isentropic_compressibility_reuss",
    "shear_modulus",
    "p_wave_velocity",
    "bulk_sound_velocity",
    "shear_wave_velocity",
    "grueneisen_parameter",
    "thermal_expansivity",
    "molar_heat_capacity_v",
    "molar_heat_capacity_p",
]


@pytest.mark.parametrize(
    "precise", [False, True], ids=["upstream-default", "tight-volume"]
)
@pytest.mark.parametrize("name", ["mgd", "mgd_modified", "solution", "assemblage"])
def test_native_material_failures_against_python(name, precise, monkeypatch):
    if precise:
        original = scipy.optimize.brentq

        def tight_volume(*args, **kwargs):
            kwargs["xtol"] = 1e-24
            return original(*args, **kwargs)

        monkeypatch.setattr(scipy.optimize, "brentq", tight_volume)
    reference = {
        "mgd": mgd,
        "mgd_modified": lambda: mgd(True),
        "solution": bridgmanite,
        "assemblage": pyrolite,
    }[name]()
    if isinstance(reference, burnman.Composite):
        native = bm.Assemblage(
            [from_burnman(p) for p in reference.phases], reference.molar_fractions
        )
    else:
        native = from_burnman(reference)
    pressure, temperature = {
        "mgd": (55e9, 1000.0),
        "mgd_modified": (55e9, 1000.0),
        "solution": (40e9, 2000.0),
        "assemblage": (50e9, 2000.0),
    }[name]
    reference.set_state(pressure, temperature)
    native.set_state(pressure, temperature)
    # Default scipy volume tolerances introduce errors of several parts in
    # 1e8 in derived energies. A tighter root gives independent EOS agreement.
    tolerance = 3e-11 if precise else 5e-8
    for prop in PROPERTIES:
        reference_prop = (
            "effective_shear_modulus"
            if name == "assemblage" and prop == "shear_modulus"
            else prop
        )
        assert getattr(native, prop) == pytest.approx(
            getattr(reference, reference_prop), rel=tolerance, abs=1e-20
        ), f"{name}: {prop}"
    if name == "solution":
        np.testing.assert_allclose(
            native.partial_entropies, reference.partial_entropies, rtol=tolerance
        )
        np.testing.assert_allclose(
            native.partial_gibbs, reference.partial_gibbs, rtol=tolerance
        )
        # Ideal enthalpy cancels G and TS; the input scale sets roundoff.
        error = 8 * np.finfo(float).eps * abs(reference.excess_gibbs)
        assert native.excess_enthalpy == pytest.approx(
            reference.excess_enthalpy, abs=error
        )


@pytest.mark.parametrize("method", ["bm3", "vinet"])
def test_reference_energy_with_validated_defaults(method):
    params = dict(
        equation_of_state=method,
        V_0=11.24e-6,
        K_0=161e9,
        Kprime_0=3.8,
        molar_mass=0.0403,
        n=2,
    )
    reference = burnman.Mineral(params.copy())
    native = bm.Mineral(params.copy())
    reference.set_state(0.0, 300.0)
    native.set_state(0.0, 300.0)
    # Upstream calls the isothermal reference energy F_0; C++ calls it E_0.
    assert reference.params["F_0"] == native.params.E_0 == 0.0
    # At V_0 the energy is exactly E_0; solved volumes may differ by roundoff.
    assert native.molar_internal_energy == pytest.approx(
        reference.molar_internal_energy, abs=1e-9
    )
    assert native.molar_gibbs == pytest.approx(reference.molar_gibbs, abs=1e-9)


def test_bragg_williams_pressure_derivative_roundoff():
    params = dict(n=1, factor=0.8, deltaH=1000.0, deltaV=1e-7, Wh=1000.0, Wv=1e-7)
    reference, state = property_modifiers.bragg_williams_excesses(5e10, 2000.0, params)
    mineral = bm.Mineral(mgd().params)
    mineral.set_property_modifiers([["bragg_williams", params]])
    mineral.set_state(5e10, 2000.0)
    native = mineral.get_property_modifiers()
    assert state["Q"] == 0.0
    error = 8 * np.finfo(float).eps * abs(reference["G"]) / 1000.0
    assert native["dGdP"] == pytest.approx(reference["dGdP"], abs=error)
    assert native["dGdP"] == pytest.approx(params["deltaV"], abs=error)


def test_modified_tait_zero_pressure_cancellation():
    params = dict(
        equation_of_state="mt",
        V_0=2.445e-5,
        K_0=251e9,
        Kprime_0=4.14,
        Kdprime_0=-1.6e-11,
        molar_mass=0.1,
        n=5,
    )
    reference = burnman.Mineral(params.copy())
    native = bm.Mineral(params.copy())
    reference.set_state(0.0, 300.0)
    native.set_state(0.0, 300.0)
    error = 8 * np.finfo(float).eps * reference.params["V_0"] * reference.params["P_0"]
    assert native.molar_gibbs == pytest.approx(reference.molar_gibbs, abs=error)


def partitioning_assemblage():
    phases = []
    for name, shifts, fractions in [
        ("Mg-rich", (0.0, 10000.0), [0.8, 0.2]),
        ("Fe-rich", (10000.0, 0.0), [0.2, 0.8]),
    ]:
        endmembers = []
        for element, shift in zip(["Mg", "Fe"], shifts):
            params = dict(fixture_mineral("periclase").params)
            params.update(name=f"{element}O", formula={element: 1.0, "O": 1.0})
            params["F_0"] += shift
            if element == "Fe":
                params["molar_mass"] = 0.0718444
            endmembers.append((burnman.Mineral(params), f"[{element}]O"))
        phases.append(
            burnman.Solution(
                name=name,
                solution_model=models.IdealSolution(endmembers),
                molar_fractions=fractions,
            )
        )
    reference = burnman.Composite(phases, [0.4, 0.6])
    reference.number_of_moles = 1.0
    reference.set_state(1e9, 2000.0)
    native = bm.Assemblage([from_burnman(p) for p in phases], [0.4, 0.6])
    native.set_state(1e9, 2000.0)
    return reference, native


@pytest.mark.parametrize("temperature", [1500.0, 2000.0])
def test_equilibrate_objective_and_jacobian_against_main(temperature):
    reference, native = partitioning_assemblage()
    bulk = {"Mg": 0.5, "Fe": 0.5, "O": 1.0}
    constraints = [("P", 1e9), ("T", temperature)]
    expected, parameters = burnman.equilibrate(
        bulk, reference, list(constraints), store_assemblage=False
    )
    assert expected.success
    actual = bm.equilibrate(
        bulk,
        native,
        [bm.PressureConstraint(1e9), bm.TemperatureConstraint(temperature)],
        tol=1e-8,
    )
    solve = actual.sol_array.item()
    assert solve.success, solve.message
    scales = np.array(
        [equilibration.P_scaling, equilibration.T_scaling, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(solve.x, expected.x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        bm.get_parameter_vector(native),
        equilibration.get_parameters(reference) * scales,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        bm.get_endmember_amounts(native),
        equilibration.get_endmember_amounts(reference),
        rtol=1e-8,
    )
    rows = np.ones(6)
    rows[:2] = scales[:2]
    rows[2 : 2 + reference.n_reactions] = equilibration.G_scaling
    values = solve.x / scales
    residual = equilibration.F(
        values,
        reference,
        constraints,
        parameters.reduced_composition_vector,
        parameters.reduced_free_composition_vectors,
    )
    np.testing.assert_allclose(solve.F, residual * rows, atol=1e-7)
    jacobian = equilibration.jacobian(
        values, reference, constraints, parameters.reduced_free_composition_vectors
    )
    np.testing.assert_allclose(
        solve.J, rows[:, None] * jacobian / scales[None, :], rtol=3e-10, atol=1e-8
    )
