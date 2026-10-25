"""DKS liquid/solid reference comparisons, derivatives and optional magnetism."""

import importlib
import inspect

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman

burnman = pytest.importorskip("burnman")


@pytest.fixture(autouse=True)
def resolve_reference_volumes(monkeypatch):
    # Upstream uses scipy.brentq's default 2e-12 m^3/mol tolerance. Resolve
    # the same roots more accurately to isolate thermodynamic differences.
    from scipy import optimize

    original = optimize.brentq

    def brentq(*args, **kwargs):
        kwargs.setdefault("xtol", 1.0e-20)
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize, "brentq", brentq)


def entries(dataset):
    module = importlib.import_module(f"burnman.minerals.{dataset}")
    return [
        (name, value)
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, burnman.Mineral)
        and value.__module__ == module.__name__
    ]


ENTRIES = [
    (dataset, name, cls)
    for dataset in ("DKS_2013_solids", "DKS_2013_liquids")
    for name, cls in entries(dataset)
]
STATES = [
    (0.0, 3000.0),
    (5.0e9, 3000.0),
    (25.0e9, 3000.0),
    (50.0e9, 4000.0),
    (100.0e9, 6000.0),
    (136.0e9, 6000.0),
]
PROPERTIES = [
    "molar_volume",
    "molar_helmholtz",
    "molar_gibbs",
    "molar_entropy",
    "molar_internal_energy",
    "molar_enthalpy",
    "molar_heat_capacity_v",
    "molar_heat_capacity_p",
    "thermal_expansivity",
    "grueneisen_parameter",
    "isothermal_bulk_modulus_reuss",
    "isentropic_bulk_modulus_reuss",
]


@pytest.mark.parametrize(
    "dataset,name,cls", ENTRIES, ids=[f"{d}.{n}" for d, n, _ in ENTRIES]
)
@pytest.mark.parametrize("pressure,temperature", STATES)
def test_dks_catalogues_match_python(dataset, name, cls, pressure, temperature):
    reference = cls()
    native = getattr(getattr(bm.minerals, dataset), name)()
    for phase in (native, reference):
        phase.set_state(pressure, temperature)
    for prop in PROPERTIES:
        assert getattr(native, prop) == pytest.approx(
            getattr(reference, prop), rel=2.0e-11, abs=1.0e-8
        ), prop
    assert native.params.equation_of_state == bm.eos_type(
        reference.params["equation_of_state"]
    )
    if dataset.endswith("liquids"):
        assert native.shear_modulus == 0.0
        np.testing.assert_array_equal(native.params.a, reference.params["a"])
    else:
        assert np.isnan(native.shear_modulus)


@pytest.mark.parametrize("dataset,name,cls", ENTRIES)
def test_dks_adapter_and_independent_factories(dataset, name, cls):
    reference = cls()
    native = from_burnman(reference)
    factory = getattr(getattr(bm.minerals, dataset), name)
    independent = factory()
    native.set_state(100.0e9, 5000.0)
    independent.set_state(0.0, 3000.0)
    reference.set_state(100.0e9, 5000.0)
    assert native.molar_gibbs == pytest.approx(reference.molar_gibbs, rel=2.0e-9)
    assert native.molar_volume != independent.molar_volume


@pytest.mark.parametrize("temperature", [1500.0, 3000.0, 6000.0])
def test_dks_magnetic_liquid(temperature):
    # The DKS liquid implementation also supplies the optional magnetic
    # extension used by Ramo and Stixrude's liquid fayalite.
    reference = burnman.minerals.RS_2014_liquids.Fe2SiO4_liquid()
    native = from_burnman(reference)
    for phase in (native, reference):
        phase.set_state(50.0e9, temperature)
    for prop in PROPERTIES:
        assert getattr(native, prop) == pytest.approx(
            getattr(reference, prop), rel=2.0e-11, abs=1.0e-8
        ), prop


@pytest.mark.parametrize("reference_name", ["periclase", "perovskite", "stishovite"])
def test_dks_solid_shear_and_zero_grueneisen(reference_name):
    reference = getattr(burnman.minerals.DKS_2013_solids, reference_name)()
    reference.params.update(
        G_0=100.0e9, Gprime_0=1.5, eta_s_0=2.0, power_law_grueneisen=False
    )
    native = from_burnman(reference)
    assert native.params.power_law_grueneisen is False
    for phase in (native, reference):
        phase.set_state(50.0e9, 4000.0)
    assert native.shear_modulus == pytest.approx(reference.shear_modulus, rel=2.0e-9)
    reference.params["grueneisen_0"] = 0.0
    native = bm.Mineral(reference.params)
    native.set_state(50.0e9, 4000.0)
    assert native.grueneisen_parameter == 0.0
    assert np.isfinite(native.isothermal_bulk_modulus_reuss)


def test_dks_parameter_aliases_and_electronic_onset():
    p = dict(burnman.minerals.DKS_2013_liquids.MgO_liquid().params)
    p.pop("el_V_0")
    native = bm.Mineral(p)
    assert native.params.el_V_0 == native.params.V_0
    assert native.params.order_f == p["O_f"]
    assert native.params.order_theta == p["O_theta"]
    with pytest.raises(ValueError, match="more than once"):
        bm.mineral_params(O_f=3, order_f=3)
    reference = burnman.eos.dks_liquid.DKS_L()
    reference.validate_parameters(p)
    v = p["V_0"]
    for t in [p["Tel_0"] - 1.0, p["Tel_0"], p["Tel_0"] + 1.0]:
        pressure = reference.pressure(t, v, p)
        eos = bm.make_eos("dks_l")
        assert eos.compute_pressure(t, v, native.params) == pytest.approx(
            pressure, abs=1.0e-4
        )
        assert eos.compute_molar_heat_capacity_v(
            pressure, t, v, native.params
        ) == pytest.approx(
            reference._molar_heat_capacity_v(pressure, t, v, p), rel=2.0e-12
        )
