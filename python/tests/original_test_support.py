# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Argument and assertion adapters for the standalone upstream test ports.

All material, EOS and fitting calculations use burnman_cpp. These helpers only
translate parameter snapshots, constructors and property names; they do not
implement missing BurnMan algorithms.
"""

import copy
import random
from functools import partial
from types import SimpleNamespace
from collections.abc import Mapping
import unittest

import numpy as np

import burnman_cpp as burnman


class BurnManTest(unittest.TestCase):
    """The original test suite's relative/absolute assertions, unchanged."""

    def run(self, result=None):
        import matplotlib.pyplot as plt

        numpy_state, random_state = np.random.get_state(), random.getstate()
        figures = set(plt.get_fignums())
        try:
            with plt.rc_context():
                return super().run(result)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(random_state)
            for figure in set(plt.get_fignums()) - figures:
                plt.close(figure)

    def assertFloatEqual(self, a, b, tol=1e-5, tol_zero=1e-16):
        self.assertAlmostEqual(a, b, delta=max(tol_zero, max(abs(a), abs(b)) * tol))

    def assertArraysAlmostEqual(self, a, b, tol=1e-5, tol_zero=1e-16):
        self.assertEqual(len(a), len(b))
        for i1, i2 in zip(a, b):
            self.assertFloatEqual(i1, i2, tol, tol_zero)


def parameter_dict(params):
    """Express a native parameter snapshot using the original dictionary keys."""
    if not isinstance(params, burnman.MineralParams):
        return params
    values = {
        key: getattr(params, key)
        for key in dir(params)
        if not key.startswith("_") and getattr(params, key) is not None
    }
    for original, native in (("n", "napfu"), ("Debye_0", "debye_0")):
        if native in values:
            values[original] = values.pop(native)
    values.pop("O_f", None)
    values.pop("O_theta", None)
    method = values.get("equation_of_state")
    if method is not None:
        values["equation_of_state"] = next(
            name for name, kind in burnman._EOS_NAMES.items() if kind == method
        )
    if params.Cp is not None:
        values["Cp"] = [getattr(params.Cp, key) for key in "abcd"]
    if method == burnman.EOSType.AA:
        for key in ("a", "b", "Theta", "lmda", "F"):
            values[key] = values.pop("aa_" + key)
        for key, native in (
            ("K_S", "K_0"),
            ("Kprime_S", "Kprime_0"),
            ("Kprime_prime_S", "Kdprime_0"),
        ):
            values[key] = values.pop(native)
    elif method == burnman.EOSType.BroshCalphad:
        values["a"] = values.pop("brosh_a")
        values["c"] = values.pop("brosh_c")
        for key, pair in (("b", ("b_0", "b_1")), ("delta", ("delta_0", "delta_1"))):
            values[key] = [values.pop(field) for field in pair]
    if method == burnman.EOSType.HPTMT and params.H_0 is not None:
        values.pop("E_0", None)
    return values


def set_parameter(material, key, value):
    """Apply a parameter edit through the native validated snapshot setter."""
    if not isinstance(material.params, burnman.MineralParams):
        material.params[key] = value
        return
    params = parameter_dict(material.params)
    params[key] = value
    material.set_parameters(burnman.mineral_params(params))


def native_copy(value):
    return value.copy() if isinstance(value, burnman.Material) else copy.deepcopy(value)


def make_assemblage(phases, fractions=None, fraction_type="molar", name=None):
    if fractions is None:
        fractions = np.full(len(phases), 1.0 / len(phases))
    if isinstance(fraction_type, str):
        fraction_type = getattr(burnman.FractionType, fraction_type.title())
    result = burnman.Assemblage(phases, fractions, fraction_type)
    if name is not None:
        result.set_name(name)
    return result


def mineral_eos(material):
    return burnman.make_eos(material.params.equation_of_state)


def native_method(method):
    return (
        method
        if isinstance(method, burnman.EquationOfState)
        else burnman.eos_type(method)
    )


def make_phase(factory, fractions=None):
    phase = factory()
    if fractions is not None:
        phase.set_composition(fractions)
    return phase


def native_endmembers(material):
    return (
        material.solution_model.endmembers
        if isinstance(material, burnman.Solution)
        else material.endmembers
    )


def eos_call(eos, method, *args):
    """Use the native EOS's compute_* names and typed final parameter argument."""
    aliases = {
        "gibbs_energy": "gibbs_free_energy",
        "helmholtz_energy": "helmholtz_free_energy",
    }
    values = list(args)
    if values:
        values[-1] = burnman.mineral_params(values[-1])
    name = (
        method
        if method == "validate_parameters"
        else "compute_" + aliases.get(method, method)
    )
    return getattr(eos, name)(*values)


def eos_function(eos, method):
    return partial(eos_call, eos, method)


def thermal_call(component, method, *args):
    values = list(args)
    if values and isinstance(values[-1], Mapping):
        values[-1] = burnman.mineral_params(values[-1])
    elif method.startswith("nondimensional_"):
        values.append(burnman.MineralParams())
    return getattr(component, method)(*values)


def ideal_model_call(model, method, *args):
    """Python's explicit base-class call bypasses native virtual dispatch."""
    ideal = burnman.IdealSolution(model.endmembers)
    return getattr(ideal, method)(*args)


def native_evaluate(material, names, pressures, temperatures):
    # The wave modulus K_eff differs from the thermodynamic Reuss modulus.
    # Read the same native effective modulus via K_eff = rho*v_phi**2.
    translated = ["bulk_sound_velocity" if name == "K_eff" else name for name in names]
    values = material.evaluate(translated, pressures, temperatures)
    if "K_eff" in names:
        density = material.evaluate(["density"], pressures, temperatures)[0]
        for i, name in enumerate(names):
            if name == "K_eff":
                values[i] = density * values[i] ** 2
    return values


def is_native_factory(value, kind=burnman.Material):
    module = getattr(value, "__module__", "")
    return (
        callable(value)
        and module.startswith(("burnman_cpp.minerals.", "burnman_cpp._core.minerals."))
        and isinstance(value(), kind)
    )


def equilibrate(
    composition, assemblage, constraints, free_compositional_vectors=(), **kwargs
):
    """Translate the original tuple constraints and restore native grid snapshots."""
    prm = burnman.get_equilibration_parameters(
        assemblage, composition, list(free_compositional_vectors)
    )
    groups = []
    for kind, value in constraints:
        if kind in ("P", "T", "S", "V"):
            constructor = {
                "P": burnman.PressureConstraint,
                "T": burnman.TemperatureConstraint,
                "S": burnman.EntropyConstraint,
                "V": burnman.VolumeConstraint,
            }[kind]
            groups.append([constructor(v) for v in np.atleast_1d(value)])
        elif kind == "phase_fraction":
            phase, fractions = value
            index = assemblage.phases.index(phase)
            groups.append(
                [
                    burnman.PhaseFractionConstraint(index, v, prm)
                    for v in np.atleast_1d(fractions)
                ]
            )
        elif kind == "phase_composition":
            phase, (sites, numerator, denominator, fractions) = value
            index = assemblage.phases.index(phase)
            groups.append(
                [
                    burnman.PhaseCompositionConstraint(
                        index, sites, numerator, denominator, v, assemblage, prm
                    )
                    for v in np.atleast_1d(fractions)
                ]
            )
        elif kind == "X":
            groups.append([burnman.LinearXConstraint(*value)])
        else:
            raise ValueError(f"Unsupported original constraint {kind!r}")
    result = burnman.equilibrate(
        composition,
        assemblage,
        groups,
        free_compositional_vectors=free_compositional_vectors,
        **kwargs,
    )
    converted = np.empty(result.sol_array.shape, dtype=object)
    for index in np.ndindex(converted.shape):
        solve = result.sol_array[index]
        snapshot = assemblage.copy()
        burnman.set_composition_and_state_from_parameters(snapshot, solve.x)
        converted[index] = SimpleNamespace(
            **{
                name: getattr(solve, name)
                for name in (
                    "success",
                    "code",
                    "message",
                    "x",
                    "F",
                    "J",
                    "F_norm",
                    "iteration_history",
                )
            },
            assemblage=snapshot,
        )
    return (
        converted.item() if converted.size == 1 else converted.squeeze()
    ), result.prm
