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
    Mineral,
    MineralParams,
    Solution,
    SymmetricRegularSolution,
)


def from_burnman(phase):
    """Copy a BurnMan mineral or ideal/regular solution into a native object.

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
        # BurnMan uses floating point atom counts even for integral formulae.
        if "n" in params:
            n = params["n"]
            if n != int(n):
                raise ValueError(
                    "The native EOS requires an integral number of atoms per formula unit."
                )
            params["n"] = int(n)
        # These optional fields describe provenance, not the EOS.
        params.pop("param_uncertainties", None)
        params.pop("property_modifiers", None)
        known = MineralParams()
        for key in params:
            if not hasattr(known, {"n": "napfu", "Debye_0": "debye_0"}.get(key, key)):
                raise ValueError(f"Unsupported parameter {key!r} in {phase.name!r}.")
        native = Mineral(params)
        native.set_property_modifiers(phase.property_modifiers)
        native.name = phase.name
    else:
        raise TypeError("Expected a BurnMan Mineral or Solution.")
    return native
