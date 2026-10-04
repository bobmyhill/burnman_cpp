"""BurnMan C++ mineral physics, solid solutions and equilibrium calculations.

All quantities use SI units: Pa, K, m³/mol, kg/mol and J/mol.
"""

from collections.abc import Mapping

from . import _core
from ._core import (
    AsymmetricRegularSolution,
    Assemblage,
    AveragingType,
    CompositeMaterial,
    Composition,
    CombinedMineral,
    CorkParams,
    CpParams,
    DampedNewtonResult,
    EOSType,
    EqualityConstraint,
    EquilibrateResult,
    EquilibrationParameters,
    EquationOfState,
    EntropyConstraint,
    FractionType,
    IdealSolution,
    IterationHistory,
    LinearXConstraint,
    Material,
    MaterialPolytope,
    MineralParams,
    NotImplementedError,
    PhaseCompositionConstraint,
    PhaseFractionConstraint,
    PressureConstraint,
    PTEllipseConstraint,
    Solution,
    SolutionModel,
    SymmetricRegularSolution,
    TemperatureConstraint,
    VolumeConstraint,
    file_to_composition_list,
    get_endmember_amounts,
    get_equilibration_parameters,
    get_parameter_vector,
    set_composition_and_state_from_parameters,
    composite_polytope_at_constrained_composition,
    solution_polytope_from_endmember_occupancies,
    simplify_composite_with_composition,
    transform_solution_to_new_basis,
)

__version__ = _core.__version__

_EOS_NAMES = {
    "auto": EOSType.Auto,
    "vinet": EOSType.Vinet,
    "mt": EOSType.MT,
    "bm2": EOSType.BM2,
    "bm3": EOSType.BM3,
    "mgd2": EOSType.MGD2,
    "mgd3": EOSType.MGD3,
    "hp_tmt": EOSType.HPTMT,
    "hptmt": EOSType.HPTMT,
    "hp_tmtl": EOSType.HPTMTL,
    "hptmtl": EOSType.HPTMTL,
    "slb2": EOSType.SLB2,
    "slb3": EOSType.SLB3,
    "slb3-conductive": EOSType.SLB3Conductive,
    "slb3conductive": EOSType.SLB3Conductive,
}


def eos_type(method):
    """Convert a BurnMan EOS name or EOSType into an EOSType."""
    if isinstance(method, EOSType):
        return method
    if not isinstance(method, str):
        raise TypeError("An EOS must be an EOSType or a string name.")
    try:
        return _EOS_NAMES[method.lower()]
    except KeyError:
        raise ValueError(f"Unknown equation of state: {method!r}") from None


def mineral_params(params=None, **kwargs):
    """Build typed parameters from a dictionary or keyword arguments.

    Accepts BurnMan's ``n`` and ``Debye_0`` as aliases for ``napfu`` and
    ``debye_0``. ``Cp`` can be CpParams or a four-element sequence.
    """
    if isinstance(params, MineralParams):
        if kwargs:
            raise TypeError("Keyword overrides require a parameter dictionary.")
        return params
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("params must be a dictionary or MineralParams.")
    values = dict(params or {})
    values.update(kwargs)
    result = MineralParams()
    aliases = {"n": "napfu", "Debye_0": "debye_0"}
    used = set()
    for key, value in values.items():
        name = aliases.get(key, key)
        if name in used:
            raise ValueError(
                f"Parameter {name!r} was supplied more than once through aliases."
            )
        used.add(name)
        if not hasattr(result, name):
            raise ValueError(f"Unknown mineral parameter: {key!r}")
        if name == "equation_of_state" and value is not None:
            value = eos_type(value)
        elif name == "Cp" and value is not None and not isinstance(value, CpParams):
            value = CpParams(*value)
        setattr(result, name, value)
    return result


class Mineral(_core.Mineral):
    """A validated C++ mineral constructed from a dictionary or MineralParams.

    Parameters are copied on construction. ``params`` returns a snapshot.
    """

    def __init__(self, params):
        super().__init__(mineral_params(params))


def make_eos(method):
    """Create a native equation of state from an enum or name, e.g. 'slb3'."""
    return _core.make_eos(eos_type(method))


def equilibrate(
    composition,
    assemblage,
    equality_constraints,
    free_compositional_vectors=(),
    *,
    tol=1.0e-3,
    store_iterates=False,
    max_iterations=100,
    verbose=False,
):
    """Find an assemblage's equilibrium state subject to equality constraints.

    Supply two constraints plus one per free compositional vector. Each item
    may be a constraint or a sequence of constraints. Sequences generate a
    Cartesian grid of solves, returned as ``result.sol_array`` (NumPy object
    array). ``result.prm`` describes the unknowns. Each solve exposes success,
    message, x, F and J, and optionally iteration_history. ``tol`` is an
    absolute Newton-step tolerance, either scalar or one value per parameter.

    The assemblage and its phases are mutated. To apply one grid result, call
    ``set_composition_and_state_from_parameters(assemblage, solve.x)``.
    No separate assemblage snapshots are stored by the native solver.
    """
    groups = [
        [constraint] if isinstance(constraint, EqualityConstraint) else list(constraint)
        for constraint in equality_constraints
    ]
    return _core._equilibrate(
        composition,
        assemblage,
        groups,
        list(free_compositional_vectors),
        tol,
        store_iterates,
        max_iterations,
        verbose,
    )


__all__ = [
    "MaterialPolytope",
    "composite_polytope_at_constrained_composition",
    "solution_polytope_from_endmember_occupancies",
    "simplify_composite_with_composition",
    "transform_solution_to_new_basis",
    "AsymmetricRegularSolution",
    "Assemblage",
    "AveragingType",
    "CompositeMaterial",
    "Composition",
    "CombinedMineral",
    "CorkParams",
    "CpParams",
    "DampedNewtonResult",
    "EOSType",
    "EqualityConstraint",
    "EquilibrateResult",
    "EquilibrationParameters",
    "EquationOfState",
    "EntropyConstraint",
    "FractionType",
    "IdealSolution",
    "IterationHistory",
    "LinearXConstraint",
    "Material",
    "Mineral",
    "MineralParams",
    "NotImplementedError",
    "PhaseCompositionConstraint",
    "PhaseFractionConstraint",
    "PressureConstraint",
    "PTEllipseConstraint",
    "Solution",
    "SolutionModel",
    "SymmetricRegularSolution",
    "TemperatureConstraint",
    "VolumeConstraint",
    "eos_type",
    "equilibrate",
    "file_to_composition_list",
    "get_endmember_amounts",
    "get_equilibration_parameters",
    "get_parameter_vector",
    "make_eos",
    "mineral_params",
    "set_composition_and_state_from_parameters",
]

from ._core import (
    pseudosection,
    refine_pseudosection,
    stable_equilibrium,
    water_fluid,
    PseudosectionSettings,
    PseudosectionResult,
    EquilibriumState,
    PhaseBoundary,
    BoundaryPoint,
    PhaseDiagramNode,
    PhaseField,
    PhaseState,
    pseudosection_field_polygons,
    PhaseFieldPolygon,
    PseudosectionPolygons,
)

__all__ += [
    "pseudosection",
    "refine_pseudosection",
    "stable_equilibrium",
    "water_fluid",
    "PseudosectionSettings",
    "PseudosectionResult",
    "EquilibriumState",
    "PhaseBoundary",
    "BoundaryPoint",
    "PhaseDiagramNode",
    "PhaseField",
    "PhaseState",
    "pseudosection_field_polygons",
    "PhaseFieldPolygon",
    "PseudosectionPolygons",
]
