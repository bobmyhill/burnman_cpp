"""Compare every exported public dataset entry with pinned Python BurnMan."""

import copy
import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp import minerals

burnman = pytest.importorskip("burnman")
DATASETS = {
    "HP11": "HP_2011_ds62",
    "HGP18": "HGP_2018_ds633",
    "MB16": "mb50NCKFMASHTO",
    "SLB11": "SLB_2011",
    "JH15": "JH_2015",
    "MP14": "mp50NCKFMASHTO",
    "SLB24": "SLB_2024",
    "IG18": "ig50NCKFMASHTOCr",
    "HPx_ds636": "HPx_ds636",
    "IG24": "ig51W24",
    "IG25": "ig51G25",
}


def public_entries(module):
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (
            isinstance(value, burnman.Mineral)
            or (
                inspect.isclass(value)
                and issubclass(value, burnman.Mineral)
                and value.__module__ == module.__name__
            )
        )
    }


ENTRIES = [
    (group, name, value)
    for group, reference in DATASETS.items()
    for name, value in public_entries(
        importlib.import_module(f"burnman.minerals.{reference}")
    ).items()
]


@pytest.fixture(autouse=True)
def ps1994_paper_gas_constant(monkeypatch):
    # Match the native PS1994 paper constant without changing solution mixing R.
    from burnman.eos import pitzer_sterner

    monkeypatch.setattr(
        pitzer_sterner, "constants", SimpleNamespace(gas_constant=8.314510)
    )


def reference_phase(value):
    phase = value() if inspect.isclass(value) else copy.deepcopy(value)
    # Retain the existing, documented correction to HGP pseudo-species pjd=1.
    # All thermodynamic parameters and interaction matrices remain upstream's.
    if isinstance(phase, burnman.Solution):
        model = phase.solution_model
        members = [
            (m, sites.replace("[Alsi2]", "[Alsitwo]")) for m, sites in model.endmembers
        ]
        if any(old[1] != new[1] for old, new in zip(model.endmembers, members)):
            alphas = np.asarray(model.alphas)

            def interactions(matrix):
                raw = matrix * (alphas[:, None] + alphas[None, :]) / 2.0
                return [raw[i, i + 1 :].tolist() for i in range(len(members) - 1)]

            phase = burnman.Solution(
                name=phase.name,
                solution_model=burnman.classes.solutionmodel.AsymmetricRegularSolution(
                    members,
                    alphas,
                    interactions(model.We),
                    interactions(model.Wv),
                    interactions(model.Ws),
                ),
            )
    return phase


def finite_differences(phase, kind):
    if isinstance(phase, burnman.CombinedMineral):
        return any(finite_differences(m, kind) for m, _ in phase.mixture.endmembers)
    if isinstance(phase, burnman.Solution):
        return any(finite_differences(m, kind) for m, _ in phase.endmembers)
    method = {"liquid": "hp_tmtL", "water": "pitzer-sterner"}.get(kind)
    return (
        phase.params["equation_of_state"] == method
        if method
        else any(k == "bragg_williams" for k, _ in phase.property_modifiers)
    )


@pytest.mark.parametrize("group", ["HGP18", "HPx_ds636"])
@pytest.mark.parametrize("pressure", [1.0e5, 3.0e9])
@pytest.mark.parametrize("temperature", [700.0, 1200.0, 1800.0])
@pytest.mark.parametrize("factor", [0.8, -1.0, -0.5])
@pytest.mark.parametrize("n", [1.0, 3.0])
def test_ordering_derivatives_match_python_gibbs(
    group, pressure, temperature, factor, n
):
    from burnman.eos.property_modifiers import bragg_williams_excesses

    reference = getattr(burnman.minerals, DATASETS[group]).san()
    native = getattr(minerals, group).san()
    params = reference.property_modifiers[0][1]
    params["factor"] = factor
    params["n"] = n
    native.set_property_modifiers(reference.property_modifiers)
    native.set_state(pressure, temperature)

    def gibbs(dp=0.0, dt=0.0):
        return bragg_williams_excesses(pressure + dp, temperature + dt, params)[0]["G"]

    values = native.get_property_modifiers()
    g = gibbs()
    assert values["G"] == pytest.approx(g, rel=1.0e-12)
    # Wider stencils resolve curvature lost by the upstream 1 kPa stencil;
    # five-point differences suppress truncation error in the wider steps.
    dp, dt = 1.0e6, 0.5
    roundoff = 16.0 * np.finfo(float).eps * abs(g)
    for axis, step in [("P", dp), ("T", dt)]:
        energies = [
            gibbs(dp=offset) if axis == "P" else gibbs(dt=offset)
            for offset in [-2.0 * step, -step, step, 2.0 * step]
        ]
        first = (energies[0] - 8.0 * energies[1] + 8.0 * energies[2] - energies[3]) / (
            12.0 * step
        )
        second = (
            -energies[0]
            + 16.0 * energies[1]
            - 30.0 * g
            + 16.0 * energies[2]
            - energies[3]
        ) / (12.0 * step**2)
        assert values[f"dGd{axis}"] == pytest.approx(
            first, rel=1.0e-8, abs=roundoff / step
        )
        assert values[f"d2Gd{axis}2"] == pytest.approx(
            second, rel=1.0e-3, abs=roundoff / step**2
        )
    mixed = (gibbs(dp, dt) - gibbs(dp, -dt) - gibbs(-dp, dt) + gibbs(-dp, -dt)) / (
        4.0 * dp * dt
    )
    assert values["d2GdPdT"] == pytest.approx(
        mixed, rel=1.0e-3, abs=roundoff / (dp * dt)
    )


@pytest.mark.parametrize("group", ["IG18", "IG24", "IG25"])
def test_kjd_bulk_modulus_matches_python_volume_derivative(group):
    reference = copy.deepcopy(getattr(burnman.minerals, DATASETS[group]).kjd)
    native = getattr(minerals, group).kjd()
    pressure, temperature, step = 3.0e9, 1200.0, 1.0e6
    reference.set_state(pressure - step, temperature)
    lower = reference.molar_volume
    reference.set_state(pressure + step, temperature)
    upper = reference.molar_volume
    reference.set_state(pressure, temperature)
    native.set_state(pressure, temperature)
    expected = -reference.molar_volume * 2.0 * step / (upper - lower)
    assert native.isothermal_bulk_modulus_reuss == pytest.approx(expected, rel=1.0e-5)


@pytest.mark.parametrize("group", DATASETS)
def test_complete_public_inventory(group):
    expected = public_entries(
        importlib.import_module(f"burnman.minerals.{DATASETS[group]}")
    )
    actual = importlib.import_module(f"burnman_cpp.minerals.{group}")
    assert actual is getattr(minerals, group)
    assert all(hasattr(actual, name) for name in expected)


@pytest.mark.parametrize(
    "group,name,value", ENTRIES, ids=[f"{g}.{n}" for g, n, _ in ENTRIES]
)
@pytest.mark.parametrize("state", [0, 1], ids=["cool", "hot"])
def test_every_factory_matches_python_burnman(group, name, value, state):
    native = getattr(getattr(minerals, group), name)()
    reference = reference_phase(value)
    if isinstance(reference, burnman.Solution):
        assert native.n_endmembers == reference.n_endmembers
        assert native.endmember_names == reference.endmember_names
        if isinstance(reference, burnman.RelaxedSolution):
            assert isinstance(native, bm.RelaxedSolution)
            np.testing.assert_array_equal(native.dndq, reference.dndq)
            np.testing.assert_array_equal(native.dndx, reference.dndx)
            n = reference.n_unrelaxed_vectors
        else:
            n = reference.n_endmembers
            np.testing.assert_allclose(
                native.endmember_n_occupancies,
                reference.solution_model.endmember_noccupancies,
                atol=1.0e-14,
                rtol=0.0,
            )
        fractions = np.arange(1.0, n + 1)
        fractions /= fractions.sum()
        for phase in (native, reference):
            phase.set_composition(fractions)
    pressure, temperature = (
        [(10.0e9, 800.0), (80.0e9, 2000.0)]
        if group.startswith("SLB")
        else [(1.0e5, 700.0), (3.0e9, 1200.0)]
    )[state]
    for phase in (native, reference):
        phase.set_state(pressure, temperature)
    assert native.name == reference.name
    for element in native.formula.keys() | reference.formula.keys():
        assert native.formula.get(element, 0.0) == pytest.approx(
            reference.formula.get(element, 0.0), abs=1.0e-12
        )
    # SciPy's upstream Nelder-Mead spin minima have finite composition accuracy.
    relaxed = isinstance(reference, burnman.RelaxedSolution)
    rtol = 3.0e-5 if relaxed else 8.0e-8
    bw = finite_differences(reference, "ordering")
    liquid = finite_differences(reference, "liquid")
    water = finite_differences(reference, "water")
    if relaxed:
        np.testing.assert_allclose(
            native.molar_fractions, reference.molar_fractions, rtol=0.0, atol=1.0e-5
        )
    for prop in (
        "molar_mass",
        "molar_gibbs",
        "molar_entropy",
        "molar_volume",
        "molar_enthalpy",
        "molar_heat_capacity_p",
        "molar_heat_capacity_v",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
    ):
        tolerance = rtol
        absolute = 1.0e-12
        if water and prop in (
            "molar_heat_capacity_p",
            "molar_heat_capacity_v",
            "isentropic_bulk_modulus_reuss",
            "thermal_expansivity",
        ):
            tolerance = max(tolerance, 2.0e-5)  # finite-difference water derivatives
        # Upstream differentiates BW G using dP=1000 Pa, which loses pressure
        # curvature to roundoff. Native ordering derivatives are analytic.
        # Keep G,S,V strict and allow upstream noise in derived properties.
        if bw:
            tolerance = max(
                tolerance,
                {
                    "isothermal_bulk_modulus_reuss": 0.02,
                    "isentropic_bulk_modulus_reuss": 0.02,
                    "molar_heat_capacity_v": 5.0e-4,
                    # Upstream's 0.1 K second difference has O(dT^2) error.
                    "molar_heat_capacity_p": 5.0e-7,
                    "thermal_expansivity": 2.0e-5,
                }.get(prop, rtol),
            )
        if liquid and prop in (
            "molar_heat_capacity_p",
            "molar_heat_capacity_v",
            "isentropic_bulk_modulus_reuss",
        ):
            tolerance = max(tolerance, 2.0e-6)  # numerical second T derivative
        if liquid and prop == "molar_heat_capacity_v":
            # Cv subtracts the expansion term from Cp and can approach zero.
            absolute = 2.0e-6 * max(
                abs(native.molar_heat_capacity_p), abs(reference.molar_heat_capacity_p)
            )
        if liquid and prop == "isentropic_bulk_modulus_reuss":
            # Ks = Kt Cp/Cv amplifies the same near-zero Cv cancellation.
            tolerance = max(
                tolerance,
                2.0e-6
                * (
                    1.0
                    + 2.0
                    * abs(
                        reference.molar_heat_capacity_p
                        / reference.molar_heat_capacity_v
                    )
                ),
            )
        np.testing.assert_allclose(
            getattr(native, prop),
            getattr(reference, prop),
            rtol=tolerance,
            atol=absolute,
            err_msg=f"{group}.{name}: {prop}",
        )
    if isinstance(reference, burnman.Solution) and not relaxed:
        for prop in (
            "partial_gibbs",
            "partial_entropies",
            "partial_volumes",
            "gibbs_hessian",
            "entropy_hessian",
            "volume_hessian",
        ):
            np.testing.assert_allclose(
                getattr(native, prop),
                getattr(reference, prop),
                rtol=8.0e-8,
                atol=1.0e-8,
                err_msg=prop,
            )


@pytest.mark.parametrize(
    "name,n",
    [
        ("silicate_melt", 12),
        ("CMS_melt", 3),
        ("MS_melt", 2),
    ],
)
def test_full_and_reduced_melt_inventories(name, n):
    phase = getattr(minerals.HGP18, name)()
    assert phase.n_endmembers == n
    assert ("Cr" in phase.elements) == (name == "silicate_melt")


@pytest.mark.parametrize(
    "name, composition",
    [
        ("ferropericlase_relaxed", [0.75, 0.2, 0.03, 0.02]),
        ("bridgmanite_relaxed", [0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
    ],
)
def test_relaxed_thermal_derivatives_and_state_updates(name, composition):
    phase = getattr(minerals.SLB24, name)()
    phase.set_composition(composition)
    p, t, dp, dt = 60.0e9, 1800.0, 1.0e6, 0.1
    phase.set_state(p, t)
    volume, entropy = phase.molar_volume, phase.molar_entropy
    kt, alpha, cp = (
        phase.isothermal_bulk_modulus_reuss,
        phase.thermal_expansivity,
        phase.molar_heat_capacity_p,
    )

    def properties(p, t):
        phase.set_state(p, t)
        return np.array([phase.molar_volume, phase.molar_entropy])

    d_dp = (properties(p + dp, t) - properties(p - dp, t)) / (2.0 * dp)
    d_dt = (properties(p, t + dt) - properties(p, t - dt)) / (2.0 * dt)
    assert d_dp[0] == pytest.approx(-volume / kt, rel=1.0e-5)
    assert d_dt[0] == pytest.approx(volume * alpha, rel=1.0e-5)
    assert d_dt[1] == pytest.approx(cp / t, rel=1.0e-5)
    assert d_dp[1] == pytest.approx(-volume * alpha, rel=1.0e-5)
    # The composition setter must refresh the spin state at an existing P,T.
    phase.set_state(p, t)
    phase.set_composition(np.array(composition)[::-1])
    assert phase.molar_entropy != pytest.approx(entropy)
    fresh = getattr(minerals.SLB24, name)()
    fresh.set_composition(np.array(composition)[::-1])
    fresh.set_state(p, t)
    np.testing.assert_allclose(
        phase.molar_fractions, fresh.molar_fractions, atol=1.0e-9
    )
    with pytest.raises(ValueError):
        phase.set_composition(np.ones(phase.n_endmembers) / phase.n_endmembers)


def test_relaxed_solution_owns_endmember_state():
    parent = minerals.SLB24.ferropericlase()
    template = minerals.SLB24.ferropericlase_relaxed()
    relaxed = bm.RelaxedSolution(parent, template.dndq.T, template.dndx.T)
    parent.set_state(10.0e9, 1000.0)
    expected = parent.molar_gibbs
    relaxed.set_state(80.0e9, 2000.0)
    parent.reset_cache()
    assert parent.molar_gibbs == expected


@pytest.mark.parametrize("name", ["quartz", "stishovite"])
@pytest.mark.parametrize("temperature", [300.0, 1200.0])
def test_slb2011_landau_ordered_and_disordered_entropy(name, temperature):
    native = getattr(minerals.SLB11, name)()
    reference = getattr(burnman.minerals.SLB_2011, name)()
    for phase in (native, reference):
        phase.set_state(1.0e5, temperature)
    assert native.molar_entropy == pytest.approx(reference.molar_entropy, rel=8.0e-8)
