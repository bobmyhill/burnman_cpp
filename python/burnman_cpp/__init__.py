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

"""BurnMan C++ mineral physics, solid solutions and equilibrium calculations.

All quantities use SI units: Pa, K, m³/mol, kg/mol and J/mol.
"""

from collections.abc import Mapping

from . import _core
from . import constants
from .classes import seismic, geotherm
from .classes.equilibratedcomposite import EquilibratedComposite
from . import (
    debye_temperature_models,
    anharmonic_prefactor_models,
    anharmonic_thermal_models,
)
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
    RelaxedSolution,
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
    visual_center_of_polygon,
    signed_distance_to_polygon,
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
    "spock": EOSType.SPOCK,
    "dks_s": EOSType.DKS_S,
    "dks_l": EOSType.DKS_L,
    "mgd2": EOSType.MGD2,
    "mgd3": EOSType.MGD3,
    "modular_mgd": EOSType.ModularMGD,
    "modular_mgd_with_anharmonicity": EOSType.ModularMGDWithAnharmonicity,
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
    ``debye_0``. Atom counts may be fractional. ``Cp`` accepts four coefficients.
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
    aliases = {
        "n": "napfu",
        "Debye_0": "debye_0",
        "O_theta": "order_theta",
        "O_f": "order_f",
    }
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
        elif name == "reference_eos" and isinstance(value, (str, EOSType)):
            value = make_eos(value)
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
    stagnation_iterations=0,
):
    """Find an assemblage's equilibrium state subject to equality constraints.

    Supply two constraints plus one per free compositional vector. Each item
    may be a constraint or a sequence of constraints. Sequences generate a
    Cartesian grid of solves, returned as ``result.sol_array`` (NumPy object
    array). ``result.prm`` describes the unknowns. Each solve exposes success,
    message, x, F and J, and optionally iteration_history. ``tol`` is an
    absolute Newton-step tolerance, either scalar or one value per parameter.
    ``stagnation_iterations`` optionally stops repeated negligible movements
    without an improving Newton correction, returning failure code 4. Zero
    disables this check; the successful-convergence criteria are unchanged.

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
        stagnation_iterations,
    )


__all__ = [
    "EquilibratedComposite",
    "seismic",
    "geotherm",
    "debye_temperature_models",
    "anharmonic_prefactor_models",
    "anharmonic_thermal_models",
    "constants",
    "MaterialPolytope",
    "composite_polytope_at_constrained_composition",
    "solution_polytope_from_endmember_occupancies",
    "visual_center_of_polygon",
    "signed_distance_to_polygon",
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
    "RelaxedSolution",
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
    phase_diagram,
    phase_diagram_contours,
    PhaseDiagramContourSettings,
    PhaseDiagramContourLine,
    PhaseDiagramContourResult,
    stable_equilibrium,
    water_fluid,
    PhaseDiagramSettings,
    PhaseDiagramResult,
    EquilibriumState,
    PhaseBoundary,
    BoundaryPoint,
    PhaseDiagramNode,
    PhaseField,
    PhaseState,
    phase_diagram_field_polygons,
    verify_phase_diagram_critical_endpoints,
    PhaseFieldPolygon,
    PhaseDiagramPolygons,
)
from .tools.phase_diagram import (
    calculate_phase_diagram_patch,
    merge_phase_diagrams,
    phase_diagram_field_regions,
    phase_diagram_problem_regions,
    save_phase_diagram,
    load_phase_diagram,
    plot_phase_diagram,
    plot_phase_diagram_contours,
    refine_phase_diagram,
)
from .tools.contours import phase_diagram_contour_levels
from .tools.equilibrium_path import (
    EquilibriumPath,
    calculate_equilibrium_path,
    sample_equilibrium_path,
    plot_mode_box,
)
from .tools.properties import (
    phase_diagram_grid,
    phase_diagram_grid_chunks,
    load_property_grid_chunks,
    retry_property_grid_chunks,
    save_property_grid,
    load_property_grid,
    evaluate_property_grid,
)
from ._core import (
    assemblage_properties,
    thermodynamic_property_names,
    PhaseDiagramPropertyGrid,
    GridSolution,
)

__all__ += [
    "EquilibriumPath",
    "calculate_equilibrium_path",
    "sample_equilibrium_path",
    "plot_mode_box",
    "phase_diagram_grid",
    "phase_diagram_grid_chunks",
    "load_property_grid_chunks",
    "retry_property_grid_chunks",
    "save_property_grid",
    "load_property_grid",
    "evaluate_property_grid",
    "GridSolution",
    "assemblage_properties",
    "thermodynamic_property_names",
    "PhaseDiagramPropertyGrid",
    "calculate_phase_diagram_patch",
    "merge_phase_diagrams",
    "phase_diagram_field_regions",
    "phase_diagram_problem_regions",
    "save_phase_diagram",
    "load_phase_diagram",
    "phase_diagram_contour_levels",
    "phase_diagram",
    "phase_diagram_contours",
    "PhaseDiagramContourSettings",
    "PhaseDiagramContourLine",
    "PhaseDiagramContourResult",
    "plot_phase_diagram_contours",
    "refine_phase_diagram",
    "stable_equilibrium",
    "water_fluid",
    "PhaseDiagramSettings",
    "PhaseDiagramResult",
    "EquilibriumState",
    "PhaseBoundary",
    "BoundaryPoint",
    "PhaseDiagramNode",
    "PhaseField",
    "PhaseState",
    "phase_diagram_field_polygons",
    "verify_phase_diagram_critical_endpoints",
    "PhaseFieldPolygon",
    "PhaseDiagramPolygons",
    "plot_phase_diagram",
]

from .tools.potential_grid import (
    PotentialGrid,
    load_potential_grid,
    save_potential_grid,
    infill_potential_grid,
    smooth_potential_grid,
)
from .tools.phase_diagram_workflow import (
    PhaseDiagramCalculation,
    PotentialGridCalculation,
    calculate_phase_diagram_grid,
    calculate_potential_grid,
)
from .tools.aspect import AspectTableExport, write_aspect_entropy_pressure_table

__all__ += [
    "PotentialGrid",
    "load_potential_grid",
    "save_potential_grid",
    "infill_potential_grid",
    "smooth_potential_grid",
    "PhaseDiagramCalculation",
    "calculate_phase_diagram_grid",
    "PotentialGridCalculation",
    "calculate_potential_grid",
    "AspectTableExport",
    "write_aspect_entropy_pressure_table",
]
