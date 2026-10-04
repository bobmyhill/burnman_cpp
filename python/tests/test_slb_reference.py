"""Independent Python BurnMan checks of the native SLB equations and limits.

Requires the optional reference dependency (SLB_2024, BurnMan 3.x). The
reference EOS is evaluated directly to separate volume solving from property
modifiers. Explicit tests check shared domain policy and finite limits of
removable singularities; no comparison silently accepts NaNs.
"""

from copy import deepcopy
import inspect
import warnings

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024 as SLB

reference = pytest.importorskip("burnman")
pySLB = pytest.importorskip("burnman.minerals.SLB_2024")
from burnman.eos import property_modifiers
from burnman.eos.slb import SLBDomainError
from scipy.optimize import brentq

ENDMEMBERS = sorted(
    name
    for name, factory in vars(pySLB).items()
    if inspect.isclass(factory)
    and factory.__module__ == pySLB.__name__
    and factory.__name__ == name
    and issubclass(factory, reference.Mineral)
    and not issubclass(factory, reference.Solution)
)
POINTS = [
    (0.0, 0.0),
    (50.0e9, 0.0),
    (150.0e9, 0.0),
    (0.0, 300.0),
    (0.0, 1600.0),
    (0.0, 4000.0),
    (3.0e9, 4000.0),
    (20.0e9, 1200.0),
    (80.0e9, 3000.0),
    (150.0e9, 4000.0),
]
# Absolute tolerances in each property's SI units.
PROPERTY_ATOL = {
    "molar_gibbs": 2.0e-5,
    "molar_helmholtz": 2.0e-5,
    "molar_entropy": 1.0e-9,
    "molar_enthalpy": 2.0e-5,
    "molar_internal_energy": 2.0e-5,
    "isothermal_bulk_modulus_reuss": 0.05,
    "isentropic_bulk_modulus_reuss": 0.05,
    "shear_modulus": 0.05,
    "molar_heat_capacity_p": 1.0e-9,
    "molar_heat_capacity_v": 1.0e-9,
    "thermal_expansivity": 1.0e-15,
    "grueneisen_parameter": 1.0e-11,
}


def assert_properties(native, eos, params, pressure, temperature, volume):
    native.set_state(pressure, temperature)
    assert native.molar_volume == pytest.approx(volume, rel=3.0e-11, abs=1.0e-18)
    assert native.isothermal_bulk_modulus_reuss > 0.0
    # The reference keeps some fundamental EOS functions private and computes
    # H, E and K_S at Mineral level. Use the same thermodynamic identities at
    # the independently solved volume, without its automatic root bracket.
    args = (pressure, temperature, volume, params)
    helmholtz = eos._helmholtz_energy(*args)
    entropy = eos.entropy(*args)
    cv = eos._molar_heat_capacity_v(*args)
    cp = eos.molar_heat_capacity_p(*args)
    kt = eos.isothermal_bulk_modulus_reuss(*args)
    expected = dict(
        molar_gibbs=eos.gibbs_energy(*args),
        molar_helmholtz=helmholtz,
        molar_entropy=entropy,
        molar_enthalpy=helmholtz + pressure * volume + temperature * entropy,
        molar_internal_energy=helmholtz + temperature * entropy,
        isothermal_bulk_modulus_reuss=kt,
        isentropic_bulk_modulus_reuss=kt if temperature == 0.0 else kt * cp / cv,
        shear_modulus=eos.shear_modulus(*args),
        molar_heat_capacity_p=cp,
        molar_heat_capacity_v=cv,
        thermal_expansivity=eos.thermal_expansivity(*args),
        grueneisen_parameter=eos._grueneisen_parameter(*args),
    )
    for name, absolute in PROPERTY_ATOL.items():
        actual = getattr(native, name)
        assert np.isfinite(expected[name]) and np.isfinite(actual), (
            name,
            pressure,
            temperature,
        )
        assert actual == pytest.approx(expected[name], rel=3.0e-10, abs=absolute), name


def test_full_canonical_endmember_inventory_is_compared():
    # Long descriptive aliases are not separate endmembers. The catalogue
    # contains 74 canonical pure-mineral factories in this reference version.
    assert len(ENDMEMBERS) == 74
    assert all(hasattr(SLB, name) for name in ENDMEMBERS)


@pytest.mark.parametrize("name", ENDMEMBERS)
@pytest.mark.parametrize(
    "pressure,temperature",
    [(0.0, 300.0), (0.0, 1600.0), (10.0e9, 800.0), (80.0e9, 2000.0)],
)
def test_complete_minerals_include_the_same_landau_and_magnetic_modifiers(
    name, pressure, temperature
):
    # Bare-EOS equality alone would miss an error in the zero-temperature
    # magnetic rewrite or in exporting the SLB2022 Landau parameters.
    native, pure = getattr(SLB, name)(), getattr(pySLB, name)()
    native.set_state(pressure, temperature)
    pure.set_state(pressure, temperature)
    assert native.molar_volume == pytest.approx(
        pure.molar_volume, rel=3.0e-11, abs=1.0e-18
    )
    for prop, absolute in PROPERTY_ATOL.items():
        actual, expected = getattr(native, prop), getattr(pure, prop)
        assert np.isfinite(actual) and np.isfinite(expected), prop
        assert actual == pytest.approx(expected, rel=3.0e-10, abs=absolute), prop


@pytest.mark.parametrize("name", ENDMEMBERS)
@pytest.mark.parametrize("pressure,temperature", POINTS)
def test_endmember_volumes_and_all_eos_properties_against_python(
    name, pressure, temperature
):
    original = getattr(SLB, name)()
    native = bm.Mineral(original.params)  # Isolate EOS from magnetic/Landau modifiers.
    pure = getattr(pySLB, name)()
    eos, params = pure.method, pure.params
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            volume = eos.volume(pressure, temperature, params)
        except SLBDomainError:
            native.set_state(pressure, temperature)
            with pytest.raises(ValueError, match="stable|Debye"):
                _ = native.molar_volume
            return
    if name == "coes" and pressure == 150.0e9 and temperature in [0.0, 4000.0]:
        # Both automatic solvers must recover the compressed root independently
        # confirmed by a dimensionless bracket of the Python pressure equation.
        ratio = brentq(
            lambda r: eos.pressure(temperature, r * params["V_0"], params) - pressure,
            0.48,
            0.55,
            xtol=1.0e-14,
        )
        assert volume == pytest.approx(ratio * params["V_0"], rel=3.0e-12)
    native_eos = bm.make_eos(original.params.equation_of_state)
    # Independent inverse check: a coincidentally matching root is insufficient.
    assert eos.pressure(temperature, volume, params) == pytest.approx(
        pressure, abs=0.05, rel=3.0e-12
    )
    assert native_eos.compute_pressure(
        temperature, volume, original.params
    ) == pytest.approx(pressure, abs=0.05, rel=3.0e-12)
    assert_properties(native, eos, params, pressure, temperature, volume)


@pytest.mark.parametrize("method", ["slb2", "slb3", "slb3-conductive"])
@pytest.mark.parametrize("temperature", [0.0, 1.0e-6, 300.0, 1600.0, 4000.0])
@pytest.mark.parametrize("ratio", [0.75, 1.0, 1.1])
def test_eos_orders_and_electronic_terms_at_fixed_volume(method, temperature, ratio):
    params = deepcopy(pySLB.fo().params)
    params.pop("Z", None)
    assert params["n"].is_integer()
    params["n"] = int(params["n"])  # Native EOS parameters use an integer atom count.
    params.update(equation_of_state=method, bel_0=0.004, gel=1.5)
    pure = reference.Mineral(params)
    native = bm.Mineral(params)
    volume = ratio * params["V_0"]
    pressure = pure.method.pressure(temperature, volume, pure.params)
    # This synthetic comparison uses the compressed/reference-connected branch.
    assert (
        pure.method.isothermal_bulk_modulus_reuss(
            pressure, temperature, volume, pure.params
        )
        > 0.0
    )
    eos = bm.make_eos(bm.eos_type(method))
    assert eos.compute_pressure(temperature, volume, native.params) == pytest.approx(
        pressure, rel=3.0e-12, abs=0.05
    )
    assert_properties(native, pure.method, pure.params, pressure, temperature, volume)


@pytest.mark.parametrize(
    "method,factory", [("slb2", SLB.fo), ("slb3", SLB.fo), ("slb3-conductive", SLB.fea)]
)
@pytest.mark.parametrize("pressure,temperature", [(5.0e9, 1200.0), (100.0e9, 3000.0)])
def test_gibbs_derivatives_and_maxwell_relations(
    method, factory, pressure, temperature
):
    params = factory().params
    params.equation_of_state = bm.eos_type(method)
    phase = bm.Mineral(params)

    # Five-point stencils provide an independent derivative check, rather than
    # comparing only two implementations of the same derivative expressions.
    def value(p, t, prop):
        phase.set_state(p, t)
        return getattr(phase, prop)

    def derivative(prop, axis, h):
        def evaluate(offset):
            return value(
                pressure + (offset * h if axis == "P" else 0.0),
                temperature + (offset * h if axis == "T" else 0.0),
                prop,
            )

        return (evaluate(-2) - 8.0 * evaluate(-1) + 8.0 * evaluate(1) - evaluate(2)) / (
            12.0 * h
        )

    dgdP = derivative("molar_gibbs", "P", 1.0e6)
    dgdT = derivative("molar_gibbs", "T", 1.0)
    dVdP = derivative("molar_volume", "P", 1.0e6)
    dVdT = derivative("molar_volume", "T", 1.0)
    dSdT = derivative("molar_entropy", "T", 1.0)
    dSdP = derivative("molar_entropy", "P", 1.0e6)
    phase.set_state(pressure, temperature)
    assert dgdP == pytest.approx(phase.molar_volume, rel=2.0e-8, abs=1.0e-16)
    assert -dgdT == pytest.approx(phase.molar_entropy, rel=2.0e-8, abs=1.0e-7)
    assert -phase.molar_volume / dVdP == pytest.approx(
        phase.isothermal_bulk_modulus_reuss, rel=2.0e-8
    )
    assert dVdT / phase.molar_volume == pytest.approx(
        phase.thermal_expansivity, rel=2.0e-8, abs=1.0e-14
    )
    assert temperature * dSdT == pytest.approx(
        phase.molar_heat_capacity_p, rel=2.0e-8, abs=1.0e-7
    )
    assert dVdT == pytest.approx(-dSdP, rel=2.0e-8, abs=1.0e-16)
    assert phase.molar_heat_capacity_p - phase.molar_heat_capacity_v == pytest.approx(
        phase.thermal_expansivity**2
        * phase.isothermal_bulk_modulus_reuss
        * phase.molar_volume
        * temperature,
        rel=2.0e-10,
        abs=1.0e-12,
    )


@pytest.fixture(scope="module")
def forsterite_spinodal():
    phase = pySLB.fo()
    eos, params, temperature = phase.method, phase.params, 4000.0
    ratio = brentq(
        lambda r: eos.isothermal_bulk_modulus_reuss(
            0.0, temperature, r * params["V_0"], params
        ),
        1.1,
        1.6,
        xtol=1.0e-14,
    )
    return ratio * params["V_0"], eos.pressure(
        temperature, ratio * params["V_0"], params
    )


@pytest.mark.parametrize("overpressure", [1.0e8, 1.0e6, 1.0e4, 100.0])
def test_stable_volume_approaches_python_spinodal_from_above(
    forsterite_spinodal, overpressure
):
    critical_volume, minimum_pressure = forsterite_spinodal
    pressure, temperature = minimum_pressure + overpressure, 4000.0
    pure = pySLB.fo()
    native = SLB.fo()
    native.set_state(pressure, temperature)
    volume = native.molar_volume
    expected = pure.method.volume(pressure, temperature, pure.params)
    assert volume < critical_volume
    assert volume == pytest.approx(expected, rel=2.0e-11)
    assert native.isothermal_bulk_modulus_reuss > 0.0
    assert pure.method.pressure(temperature, volume, pure.params) == pytest.approx(
        pressure, abs=0.002
    )


def test_both_reject_pressures_below_python_spinodal(forsterite_spinodal):
    _, minimum_pressure = forsterite_spinodal
    native, pure = SLB.fo(), pySLB.fo()
    pressure, temperature = minimum_pressure - 100.0, 4000.0
    native.set_state(pressure, temperature)
    with pytest.raises(ValueError, match="limiting pressure"):
        _ = native.molar_volume
    with pytest.raises(SLBDomainError, match="limiting pressure"):
        pure.method.volume(pressure, temperature, pure.params)


@pytest.mark.parametrize("pressure", [0.0, 3.0e9])
def test_bcc_iron_rejects_disconnected_expanded_branch(pressure):
    pure, native = pySLB.fea(), SLB.fea()
    eos, params, temperature = pure.method, pure.params, 4000.0
    v0 = params["V_0"]
    critical_ratio = brentq(
        lambda r: eos.isothermal_bulk_modulus_reuss(0.0, temperature, r * v0, params),
        1.3,
        1.4,
        xtol=1.0e-14,
    )
    minimum_pressure = eos.pressure(temperature, critical_ratio * v0, params)
    assert minimum_pressure > 9.0e9 > pressure
    # A second root is locally stable, but is separated from V_0 by a
    # mechanically unstable interval. Both solvers must reject this branch.
    expanded = (
        brentq(
            lambda r: eos.pressure(temperature, r * v0, params) - pressure,
            1.9,
            2.1,
            xtol=1.0e-14,
        )
        * v0
    )
    assert (
        expanded > 1.9 * v0
        and eos.isothermal_bulk_modulus_reuss(pressure, temperature, expanded, params)
        > 0.0
    )
    assert eos.isothermal_bulk_modulus_reuss(0.0, temperature, 1.5 * v0, params) < 0.0
    native.set_state(pressure, temperature)
    with pytest.raises(ValueError, match="stable EOS branch"):
        _ = native.molar_volume
    with pytest.raises(SLBDomainError, match="stable EOS branch"):
        eos.volume(pressure, temperature, params)


@pytest.mark.parametrize("name,ratio", [("fo", 2.0), ("en", 0.3), ("mgts", 2.0)])
def test_both_reject_nonreal_debye_temperatures(name, ratio):
    native, pure = getattr(SLB, name)(), getattr(pySLB, name)()
    volume = ratio * pure.params["V_0"]
    with pytest.raises(SLBDomainError, match="valid range"):
        pure.method._debye_temperature(1.0 / ratio, pure.params)
    with pytest.raises(SLBDomainError, match="valid range"):
        pure.method.pressure(1600.0, volume, pure.params)
    eos = bm.make_eos(native.params.equation_of_state)
    with pytest.raises(ValueError, match="real Debye domain"):
        eos.compute_pressure(1600.0, volume, native.params)


@pytest.mark.parametrize("temperature", [300.0, 1200.0, 3000.0])
@pytest.mark.parametrize("offset", [-1.0e6, -1.0, 0.0, 1.0, 1.0e6, 50.7e9])
def test_stishovite_softening_matches_python_and_its_continuous_limit(
    temperature, offset
):
    pressure = 51.6e9 + 11.1e6 * (temperature - 300.0) + offset
    native, pure = SLB.st(), pySLB.st()
    native.set_state(pressure, temperature)
    volume = native.molar_volume
    expected = pure.method.shear_modulus(pressure, temperature, volume, pure.params)
    assert np.isfinite(expected)
    if offset == 0.0:
        # Independently check continuity without reproducing either special case.
        below = pure.method.shear_modulus(
            pressure - 0.001, temperature, volume, pure.params
        )
        above = pure.method.shear_modulus(
            pressure + 0.001, temperature, volume, pure.params
        )
        assert expected == pytest.approx(below, abs=0.1)
        assert expected == pytest.approx(above, abs=0.1)
    assert np.isfinite(native.shear_modulus)
    assert native.shear_modulus == pytest.approx(expected, rel=3.0e-10, abs=0.1)


MAGNETIC_PARAMS = dict(
    structural_parameter=0.4,
    curie_temperature=[1043.0, 2.0e-9],
    magnetic_moment=[2.22, 3.0e-12],
)
MAGNETIC_ATOL = dict(
    G=1.0e-8,
    dGdT=1.0e-10,
    dGdP=1.0e-17,
    d2GdT2=1.0e-12,
    d2GdP2=1.0e-27,
    d2GdPdT=1.0e-20,
)


def test_finite_magnetic_rewrite_is_exactly_the_original_chs_potential():
    """Verify algebraic equality, rather than only agreement at sample points."""
    import sympy as sp

    t, tc, p = sp.symbols("T Tc p", positive=True)
    tau = t / tc
    a = sp.Rational(518, 1125) + sp.Rational(11692, 15975) * (1 / p - 1)
    c = sp.Rational(79, 140) / (p * a)
    d = sp.Rational(474, 497) * (1 / p - 1) / a
    original_f = (
        1
        - (
            sp.Rational(79, 140) / (p * tau)
            + sp.Rational(474, 497)
            * (1 / p - 1)
            * (tau**3 / 6 + tau**9 / 135 + tau**15 / 600)
        )
        / a
    )
    finite_h = t - c * tc - d * tc * (tau**4 / 6 + tau**10 / 135 + tau**16 / 600)
    assert sp.simplify(t * original_f - finite_h) == 0
    # The existing SLB2024 linear reference shift is E0 - T*S_D,
    # with E0=c*Tc*S_D. It removes the CHS constant and linear terms.
    ordered_h = finite_h + c * tc - t
    assert (
        sp.simplify(ordered_h + d * tc * (tau**4 / 6 + tau**10 / 135 + tau**16 / 600))
        == 0
    )
    assert sp.limit(ordered_h, t, 0, dir="+") == 0
    assert sp.limit(sp.diff(ordered_h, t), t, 0, dir="+") == 0


@pytest.mark.parametrize("name", ["fea", "mag", "smag", "hmag"])
@pytest.mark.parametrize(
    "tau", [0.0, 1.0e-8, 1.0e-4, 0.02, 0.5, 0.999, 1.001, 2.0, 4.0]
)
def test_slb2024_magnetic_terms_match_hefesto_ordered_reference(name, tau):
    """Check the authors' Hillert-Jarl specification independently of BurnMan.

    Source: stixrude/HeFESToRepository, hillert.f, commit
    6faa4053d3af7cea7b16c4bfee8b7773fb077f9b. The ordered reference is
    applied in HeFESTo itself and via an existing CHS + linear combination
    in BurnMan. SLB2024 Appendix A2 eq. A5 has apparently misplaced brackets.
    The exact Curie point is excluded: HeFESTo chooses the low-temperature
    Cp branch there, whereas BurnMan's existing convention chooses the high
    branch. This convention was not changed by the finite evaluation rewrite.
    """
    pure, native = getattr(pySLB, name)(), getattr(SLB, name)()
    magnetic, linear = pure.property_modifiers
    assert magnetic[0] == "magnetic_chs" and linear[0] == "linear"
    params = magnetic[1]
    p, tc = params["structural_parameter"], params["curie_temperature"][0]
    sd = linear[1]["delta_S"]
    assert params["curie_temperature"][1] == params["magnetic_moment"][1] == 0.0
    assert reference.constants.gas_constant * np.log1p(
        params["magnetic_moment"][0]
    ) == pytest.approx(sd, rel=2.0e-15)
    a = 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / p - 1.0)
    c, d = 79.0 / (140.0 * p * a), 474.0 / 497.0 * (1.0 / p - 1.0) / a
    assert linear[1]["delta_E"] == pytest.approx(c * tc * sd, rel=2.0e-15)
    if tau < 1.0:
        # Ordered-state specification from hillert.f, with no 1/T term.
        f = -d * (tau**3 / 6.0 + tau**9 / 135.0 + tau**15 / 600.0)
        df = -d * (tau**2 / 2.0 + tau**8 / 15.0 + tau**14 / 40.0)
        ddf = -d * (tau + 8.0 * tau**7 / 15.0 + 14.0 * tau**13 / 40.0)
    else:
        f = -(tau**-5 / 10.0 + tau**-15 / 315.0 + tau**-25 / 1500.0) / a - 1.0 + c / tau
        df = (tau**-6 / 2.0 + tau**-16 / 21.0 + tau**-26 / 60.0) / a - c / tau**2
        ddf = (
            -(3.0 * tau**-7 + 16.0 * tau**-17 / 21.0 + 26.0 * tau**-27 / 60.0) / a
            + 2.0 * c / tau**3
        )
    temperature = tau * tc
    expected = {
        "G": sd * temperature * f,
        "S": -sd * (f + tau * df),
        "Cp": -sd * (2.0 * tau * df + tau**2 * ddf),
    }
    pure.set_state(20.0e9, temperature)
    native.set_state(20.0e9, temperature)
    for actual in [
        property_modifiers.calculate_property_modifications(pure),
        native.get_property_modifiers(),
    ]:
        assert actual["G"] == pytest.approx(expected["G"], rel=3.0e-12, abs=1.0e-8)
        assert -actual["dGdT"] == pytest.approx(expected["S"], rel=3.0e-12, abs=1.0e-12)
        assert -temperature * actual["d2GdT2"] == pytest.approx(
            expected["Cp"], rel=3.0e-12, abs=1.0e-12
        )


@pytest.mark.parametrize("pressure", [0.0, 20.0e9])
@pytest.mark.parametrize("tau", [0.02, 0.5, 0.999, 1.0, 1.001, 2.0])
def test_magnetic_gibbs_and_all_derivatives_match_python(pressure, tau):
    temperature = tau * (
        MAGNETIC_PARAMS["curie_temperature"][0]
        + pressure * MAGNETIC_PARAMS["curie_temperature"][1]
    )
    native = SLB.fea()
    native.set_property_modifiers(
        [
            (
                "magnetic_chs",
                dict(
                    structural_parameter=MAGNETIC_PARAMS["structural_parameter"],
                    curie_T=MAGNETIC_PARAMS["curie_temperature"],
                    magnetic_moment=MAGNETIC_PARAMS["magnetic_moment"],
                ),
            )
        ]
    )
    native.set_state(pressure, temperature)
    expected = property_modifiers.magnetic_excesses_chs(
        pressure, temperature, MAGNETIC_PARAMS
    )[0]
    actual = native.get_property_modifiers()
    for key in MAGNETIC_ATOL:
        assert actual[key] == pytest.approx(
            expected[key], rel=2.0e-9, abs=MAGNETIC_ATOL[key]
        ), key


def test_zero_temperature_magnetic_limit_matches_python_free_energy_derivatives():
    native, pressure = SLB.fea(), 20.0e9
    native.set_property_modifiers(
        [
            (
                "magnetic_chs",
                dict(
                    structural_parameter=MAGNETIC_PARAMS["structural_parameter"],
                    curie_T=MAGNETIC_PARAMS["curie_temperature"],
                    magnetic_moment=MAGNETIC_PARAMS["magnetic_moment"],
                ),
            )
        ]
    )
    native.set_state(pressure, 0.0)
    actual = native.get_property_modifiers()
    assert np.isfinite(list(actual.values())).all()
    expected = property_modifiers.magnetic_excesses_chs(pressure, 0.0, MAGNETIC_PARAMS)[
        0
    ]
    assert np.isfinite(list(expected.values())).all()
    for key in MAGNETIC_ATOL:
        assert actual[key] == pytest.approx(
            expected[key], rel=2.0e-9, abs=MAGNETIC_ATOL[key]
        ), key

    # Independently verify the finite zero-temperature derivatives using the
    # free energy at small positive temperatures.
    def energy(p, t):
        return property_modifiers.magnetic_excesses_chs(p, t, MAGNETIC_PARAMS)[0]["G"]

    t, hP, hT = 0.1, 1.0e7, 0.01
    assert actual["G"] == pytest.approx(
        energy(pressure, t) - t * actual["dGdT"], abs=1.0e-8
    )
    dgdP = (energy(pressure + hP, t) - energy(pressure - hP, t)) / (2.0 * hP)
    dgdT = (energy(pressure, t + hT) - energy(pressure, t - hT)) / (2.0 * hT)
    dgpp = (
        energy(pressure + hP, t) - 2.0 * energy(pressure, t) + energy(pressure - hP, t)
    ) / hP**2
    dgpt = (
        energy(pressure + hP, t + hT)
        - energy(pressure + hP, t - hT)
        - energy(pressure - hP, t + hT)
        + energy(pressure - hP, t - hT)
    ) / (4.0 * hP * hT)
    # Remove the linear-in-T contribution to compare the zero-T P derivatives.
    assert actual["dGdP"] == pytest.approx(
        dgdP - t * actual["d2GdPdT"], rel=2.0e-7, abs=1.0e-17
    )
    assert actual["dGdT"] == pytest.approx(dgdT, rel=2.0e-9)
    assert actual["d2GdP2"] == pytest.approx(dgpp, rel=2.0e-3, abs=1.0e-25)
    assert actual["d2GdPdT"] == pytest.approx(dgpt, rel=1.0e-3, abs=1.0e-17)
    assert actual["d2GdT2"] == 0.0
