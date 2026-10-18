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

from .contours import phase_diagram_contour_levels
from .properties import (
    phase_diagram_grid,
    phase_diagram_grid_chunks,
    load_property_grid_chunks,
    retry_property_grid_chunks,
    save_property_grid,
    load_property_grid,
    evaluate_property_grid,
)
from .phase_diagram import (
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
from .phase_diagram_helpers import (
    example_parser,
    load_example,
    plot_phase_diagram_example,
    refine_saved,
    report_phase_diagram,
    run_phase_diagram_example,
    save_figure,
    write_phase_diagram,
)

__all__ = [
    "phase_diagram_grid",
    "phase_diagram_grid_chunks",
    "load_property_grid_chunks",
    "retry_property_grid_chunks",
    "save_property_grid",
    "load_property_grid",
    "evaluate_property_grid",
    "calculate_phase_diagram_patch",
    "merge_phase_diagrams",
    "phase_diagram_field_regions",
    "phase_diagram_problem_regions",
    "save_phase_diagram",
    "load_phase_diagram",
    "phase_diagram_contour_levels",
    "plot_phase_diagram",
    "plot_phase_diagram_contours",
    "refine_phase_diagram",
    "example_parser",
    "load_example",
    "plot_phase_diagram_example",
    "refine_saved",
    "report_phase_diagram",
    "run_phase_diagram_example",
    "save_figure",
    "write_phase_diagram",
]
