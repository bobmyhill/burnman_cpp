# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Copy mineral data from the optional pure Python BurnMan package.

The returned objects evaluate thermodynamics and equilibria in C++. BurnMan
is only imported when this adapter is called; it is not a runtime dependency
of the core wrapper.
"""

import numpy as np

from . import (
    AsymmetricRegularSolution,
    CombinedMineral,
    IdealSolution,
    MechanicalSolution,
    Mineral,
    MineralParams,
    Solution,
    SubregularSolution,
    SymmetricRegularSolution,
    water_fluid,
)


def from_burnman(phase):
    """Copy a BurnMan mineral or supported Gibbs solution into a native object.

    Includes property modifiers, nested combined endmembers, site formulae,
    and energy/entropy/volume interactions. Unsupported models and parameter
    fields raise errors rather than silently dropping thermodynamic data.
    Each call creates independent mineral and solution state.
    """
    import burnman
    from burnman.classes import solutionmodel as models

    if isinstance(phase, burnman.Solution):
        model = phase.solution_model
        endmembers = [(from_burnman(m), sites) for m, sites in model.endmembers]
        if isinstance(model, models.AsymmetricRegularSolution):
            alphas = np.asarray(model.alphas)

            def interactions(matrix):
                raw = np.asarray(matrix) * (alphas[:, None] + alphas[None, :]) / 2.0
                return [raw[i, i + 1 :].tolist() for i in range(len(endmembers) - 1)]

            kwargs = dict(
                energy_interaction=interactions(model.We),
                entropy_interaction=interactions(model.Ws),
                volume_interaction=interactions(model.Wv),
            )
            if isinstance(model, models.SymmetricRegularSolution):
                native_model = SymmetricRegularSolution(endmembers, **kwargs)
            else:
                native_model = AsymmetricRegularSolution(
                    endmembers, alphas.tolist(), **kwargs
                )
        elif type(model) is models.IdealSolution:
            native_model = IdealSolution(endmembers)
        elif type(model) is models.MechanicalSolution:
            native_model = MechanicalSolution(endmembers)
        elif type(model) is models.SubregularSolution:
            # Retain the full cubic tensors, including terms induced by binaries.
            def terms(tensor):
                return [
                    [*indices, tensor[tuple(indices)]]
                    for indices in np.argwhere(tensor != 0)
                ]

            zeros = [
                [[0.0, 0.0] for _ in endmembers[i + 1 :]]
                for i in range(len(endmembers) - 1)
            ]
            native_model = SubregularSolution(
                endmembers,
                zeros,
                energy_ternary_terms=terms(model.Wijke),
                volume_ternary_terms=terms(model.Wijkv),
                entropy_ternary_terms=terms(model.Wijks),
            )
        else:
            raise TypeError(
                f"Unsupported BurnMan solution model: {type(model).__name__}"
            )
        fractions = getattr(
            phase, "molar_fractions", np.full(len(endmembers), 1.0 / len(endmembers))
        )
        native = Solution(native_model, fractions, name=phase.name)
    elif isinstance(phase, burnman.CombinedMineral):
        native = CombinedMineral(
            [from_burnman(m) for m, _ in phase.mixture.endmembers],
            phase.mixture.molar_fractions,
            name=phase.name,
        )
        native.set_property_modifiers(phase.property_modifiers)
    elif isinstance(phase, burnman.Mineral):
        params = dict(phase.params)
        # These optional fields describe provenance, not the EOS.
        params.pop("param_uncertainties", None)
        params.pop("property_modifiers", None)
        # Convert modular MGD component objects to their native equivalents.
        if "reference_eos" in params:
            from . import (
                make_eos,
                debye_temperature_models,
                anharmonic_prefactor_models,
                anharmonic_thermal_models,
            )

            reference_name = type(params["reference_eos"]).__name__
            reference_names = {
                "BM2": "bm2",
                "BM3": "bm3",
                "BM4": "bm4",
                "Vinet": "vinet",
                "Murnaghan": "murnaghan",
                "Morse": "morse",
                "RKprime": "rkprime",
                "MACAW": "macaw",
                "MT": "mt",
                "SPOCK": "spock",
            }
            if reference_name not in reference_names:
                raise ValueError(f"Unsupported reference EoS: {reference_name}")
            params["reference_eos"] = make_eos(reference_names[reference_name])
            for key, module in (
                ("debye_temperature_model", debye_temperature_models),
                ("anharmonic_prefactor_model", anharmonic_prefactor_models),
                ("anharmonic_thermal_model", anharmonic_thermal_models),
            ):
                if key not in params:
                    continue
                component = params[key]
                component_name = type(component).__name__
                native_type = getattr(module, component_name, None)
                if (
                    not type(component).__module__.startswith("burnman.eos.")
                    or native_type is None
                    or component_name.endswith(("Base", "Model"))
                ):
                    raise ValueError(f"Unsupported {key}: {component_name}")
                params[key] = native_type()
        # Retain the existing paper-R convention for the bundled HP water.
        # All other PS coefficient/reference choices use the general EOS.
        use_water_factory = False
        if params["equation_of_state"] == "pitzer-sterner":
            expected = burnman.minerals.HP_2011_fluids.H2O().params
            use_water_factory = params.keys() == expected.keys() and all(
                key == "name"
                or (
                    value == expected[key]
                    if isinstance(value, (str, dict))
                    else np.allclose(value, expected[key], rtol=1.0e-12, atol=0.0)
                )
                for key, value in params.items()
            )
        if use_water_factory:
            native = water_fluid()
        else:
            known = MineralParams()
            aliases = {
                "n": "napfu",
                "Debye_0": "debye_0",
                "Kprime_prime_0": "Kdprime_0",
                "K_S": "K_0",
                "Kprime_S": "Kprime_0",
                "Kprime_prime_S": "Kdprime_0",
            }
            if params["equation_of_state"] == "aa":
                aliases.update(
                    {name: "aa_" + name for name in ("a", "b", "Theta", "lmda", "F")}
                )
            elif params["equation_of_state"] == "brosh_calphad":
                aliases.update(
                    {"a": "brosh_a", "c": "brosh_c", "b": "b_0", "delta": "delta_0"}
                )
            for key in params:
                if not hasattr(known, aliases.get(key, key)):
                    raise ValueError(
                        f"Unsupported parameter {key!r} in {phase.name!r}."
                    )
            native = Mineral(params)
        native.set_property_modifiers(phase.property_modifiers)
        native.name = phase.name
    else:
        raise TypeError("Expected a BurnMan Mineral or Solution.")
    if hasattr(phase, "uncertainties"):
        native.uncertainties = phase.uncertainties
    return native
