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

"""Native numerical tools and optional plotting helpers."""

from importlib import import_module as _import_module

_MODULES = ("eos", "seismic", "chemistry", "partitioning", "thermobarometry")
_EXPORTS = {
    "equilibrium_path": (
        "EquilibriumPath",
        "calculate_equilibrium_path",
        "sample_equilibrium_path",
        "plot_mode_box",
    ),
    "contours": ("phase_diagram_contour_levels",),
    "composition_contours": ("phase_composition_grid", "phase_composition_contours"),
    "phase_diagram_labels": ("label_phase_diagram_solutions",),
    "properties": (
        "phase_diagram_grid",
        "phase_diagram_grid_chunks",
        "load_property_grid_chunks",
        "retry_property_grid_chunks",
        "save_property_grid",
        "load_property_grid",
        "evaluate_property_grid",
    ),
    "phase_diagram": (
        "calculate_phase_diagram_patch",
        "merge_phase_diagrams",
        "phase_diagram_field_regions",
        "phase_diagram_problem_regions",
        "save_phase_diagram",
        "load_phase_diagram",
        "plot_phase_diagram",
        "plot_phase_diagram_contours",
        "refine_phase_diagram",
    ),
    "phase_diagram_helpers": (
        "example_parser",
        "load_example",
        "plot_phase_diagram_example",
        "refine_saved",
        "report_phase_diagram",
        "run_phase_diagram_example",
        "save_figure",
        "write_phase_diagram",
    ),
    "potential_grid": (
        "PotentialGrid",
        "load_potential_grid",
        "save_potential_grid",
        "infill_potential_grid",
        "smooth_potential_grid",
    ),
    "phase_diagram_workflow": (
        "PhaseDiagramCalculation",
        "PotentialGridCalculation",
        "calculate_phase_diagram_grid",
        "calculate_potential_grid",
    ),
    "aspect": (
        "AspectTableExport",
        "write_aspect_entropy_pressure_table",
    ),
}
_SYMBOLS = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = [*_MODULES, *_SYMBOLS]


def __getattr__(name):
    if name in _MODULES or name in _EXPORTS:
        value = _import_module(f".{name}", __name__)
    elif name in _SYMBOLS:
        value = getattr(_import_module(f".{_SYMBOLS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))
