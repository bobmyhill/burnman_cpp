# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Evaluate scalar property expressions on saved equilibria and boundaries."""

from collections.abc import Mapping
import json
from numbers import Integral, Real

import numpy as np

from .. import _core
from .composition_contours import (
    _arithmetic_expression,
    _check_context,
    _secondary_branch,
)
from .phase_diagram import _value
from .properties import _saved_property_grid_batches


def _phase_selection(phases, include_phases, exclude_phases):
    """Resolve candidate-name filters and retain their normalized options."""
    known = {p.name for p in phases}
    selected, options = set(range(len(phases))), {}
    for option, names in (
        ("include_phases", include_phases),
        ("exclude_phases", exclude_phases),
    ):
        if names is None:
            continue
        names = (names,) if isinstance(names, str) else tuple(names)
        if any(not isinstance(name, str) or name not in known for name in names):
            raise ValueError(f"{option} must use the original candidate model names.")
        indices = {i for i, p in enumerate(phases) if p.name in names}
        if option == "include_phases":
            selected &= indices
        else:
            selected -= indices
        if names or option == "include_phases":
            options[option] = names
    return selected, options


def _property_evaluator(
    phases,
    expression,
    *,
    phase=None,
    instance=None,
    include_phases=None,
    exclude_phases=(),
    diagram=None,
    relaxed=False,
    averaging_scheme="VRH",
    copies=3,
):
    """Restore private material snapshots; attribute access never equilibrates."""
    if not isinstance(relaxed, (bool, np.bool_)):
        raise TypeError("relaxed must be a boolean.")
    if instance is not None and (
        isinstance(instance, bool) or not isinstance(instance, Integral) or instance < 1
    ):
        raise ValueError("instance must be a positive integer, starting at 1.")
    if phase is None and instance is not None:
        raise ValueError("An instance requires a phase name.")
    selected, filters = _phase_selection(phases, include_phases, exclude_phases)
    if phase is not None and filters:
        raise ValueError(
            "include_phases and exclude_phases apply only to bulk properties; omit phase=."
        )
    candidate, branch, branch_context = None, None, None
    if phase is not None:
        if not isinstance(phase, str):
            raise TypeError("phase must be a candidate name or secondary label.")
        name, separator, copy = phase.partition(" #")
        if separator:
            if instance is not None or not copy.isdecimal() or int(copy) < 1:
                raise ValueError(
                    "Select a valid saved copy by name or instance, not both."
                )
            instance = int(copy)
        matches = [i for i, p in enumerate(phases) if p.name == name]
        if len(matches) == 1:
            candidate = matches[0]
        elif diagram is not None and instance is None:
            candidate, branch, branch_context = _secondary_branch(diagram, phase)
        else:
            raise ValueError(f"No unique candidate named {phase!r}.")
    if instance is not None and instance > copies:
        raise ValueError(f"instance cannot exceed the saved maximum ({copies}).")
    if candidate is not None and not 0 <= candidate < len(phases):
        raise ValueError(
            "The secondary-labelled diagram uses different candidate models."
        )
    target = type(phases[candidate]) if candidate is not None else _core.Assemblage
    arithmetic, attributes = _arithmetic_expression(
        expression,
        {
            name
            for name in dir(target)
            if not name.startswith("_") and not callable(getattr(target, name))
        },
        kind="property",
    )
    if relaxed and (
        phase is not None or not attributes <= set(_core.thermodynamic_property_names())
    ):
        raise ValueError(
            "relaxed=True requires a supported whole-assemblage thermodynamic property."
        )
    attributes = sorted(attributes)
    scheme = getattr(_core.AveragingType, averaging_scheme, None)
    if not isinstance(scheme, _core.AveragingType):
        raise ValueError(f"Unknown averaging scheme {averaging_scheme!r}.")
    cache = {}

    def material(state, basis=None, coefficients=None):
        original = phases[_value(state, "candidate_index")]
        composition = np.asarray(_value(state, "composition"))
        if basis is None:
            basis = _value(state, "composition_basis")
        basis = np.eye(len(composition)) if basis is None else np.asarray(basis)
        identity = basis.shape == (
            len(composition),
            len(composition),
        ) and np.array_equal(basis, np.eye(len(composition)))
        if coefficients is None:
            coefficients = (
                composition
                if identity
                else np.linalg.lstsq(basis.T, composition, rcond=None)[0]
            )
        key = (_value(state, "id"), basis.shape, basis.tobytes())
        if key not in cache:
            restored = original.copy()
            if isinstance(restored, _core.Solution) and not identity:
                restored = _core.transform_solution_to_new_basis(
                    restored, basis, np.asarray(coefficients), _value(state, "name")
                )
            cache[key] = restored
        restored = cache[key]
        if isinstance(restored, _core.Solution):
            restored.set_composition(np.asarray(coefficients))
        return restored

    def evaluate(state, *, field=0, position=None, bases=None, compositions=None):
        states = _value(state, "phases")
        if candidate is not None:
            matches = [p for p in states if _value(p, "candidate_index") == candidate]
            if branch is not None:
                matches = branch(matches, field, position)
            elif instance is not None:
                matches = [
                    p for p in matches if _value(p, "id") % copies == instance - 1
                ]
            if len(matches) > 1:
                raise TypeError(
                    f"{phase!r} has coexisting copies; select an instance or secondary label."
                )
            if not matches:
                return np.nan
            index = next(
                i
                for i, p in enumerate(states)
                if _value(p, "id") == _value(matches[0], "id")
            )
            target = material(
                states[index],
                None if bases is None else bases[index],
                None if compositions is None else compositions[index],
            )
        else:
            active = [
                i
                for i, p in enumerate(states)
                if _value(p, "candidate_index") in selected
                and _value(p, "amount") > 0.0
            ]
            if not active:
                raise ValueError(
                    "No positive phase amounts remain after phase selection."
                )
            materials = [
                material(
                    states[i],
                    None if bases is None else bases[i],
                    None if compositions is None else compositions[i],
                )
                for i in active
            ]
            amounts = np.array([_value(states[i], "amount") for i in active])
            target = _core.Assemblage(materials, amounts / amounts.sum())
            target.n_moles = float(amounts.sum())
            target.set_averaging_scheme(scheme)
        target.set_state(_value(state, "pressure"), _value(state, "temperature"))
        values = (
            _core.assemblage_properties(target, attributes, relaxed=True)
            if relaxed and attributes
            else {name: getattr(target, name) for name in attributes}
        )
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name!r} must be a scalar numeric attribute.")
            if not np.isfinite(value):
                raise ValueError(f"{name!r} is nonfinite at this state.")
            values[name] = np.float64(value)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = arithmetic(values)
        if not np.isfinite(value):
            raise ValueError(
                "The property expression is undefined or nonfinite at this state."
            )
        return float(value)

    return evaluate, branch_context, candidate


def phase_diagram_property_grid(
    source,
    phases,
    property_name,
    *,
    phase=None,
    instance=None,
    include_phases=None,
    exclude_phases=(),
    diagram=None,
    relaxed=False,
    averaging_scheme=_core.AveragingType.VRH,
):
    """Evaluate scalar property arithmetic at the retained equilibrium states.

    source accepts the same retained grid/file/dictionary/chunk inputs as
    phase_composition_grid(). Supply the original candidates in order. By
    default read Assemblage attributes. property_name accepts a name such as
    density (kg/m³), or an expression such as density/1000 or
    p_wave_velocity/shear_wave_velocity. Numbers, parentheses, unary signs and
    +, -, *, / are supported; calls, indexing and attribute access are rejected.
    Each referenced public attribute must be scalar and numeric. phase selects
    one candidate or a secondary label; coexisting copies require instance= or
    a label plus diagram=.

    Bulk properties can use include_phases (default None: all candidates) and
    exclude_phases (default empty). Names select all copies of each candidate;
    exclusions are applied after inclusion. Remaining saved mole amounts are
    renormalized, preserving their relative proportions and compositions.
    Empty selections at a node are masked with a diagnostic. Filters cannot
    be combined with the single-phase selector phase=.

    Ordinary attribute getters keep saved amounts/compositions fixed. Optional
    relaxed=True uses native relaxed responses for names accepted by
    thermodynamic_property_names(), on the retained assemblage. All names in
    an expression share its phase filters and response settings. Active bases and
    original coefficients are restored; no equilibria are solved. Candidate
    objects and source states remain unchanged. Failed or absent values are
    masked; nonnumeric attributes and methods are rejected. Undefined arithmetic
    (including division by zero) is masked with a diagnostic.

    Return a scalar PhaseDiagramPropertyGrid with equilibrium_solves=0. Save
    using save_property_grid(), then contour with phase_diagram_grid_contours().
    Property/phase/response choices are retained in its quantity key for replay
    of pre-existing boundary states. Properties use the saved physical P,T;
    volume-fraction reference options do not affect property grids.
    """
    phases = list(phases)
    if not isinstance(averaging_scheme, _core.AveragingType):
        raise TypeError("averaging_scheme must be an AveragingType.")
    if isinstance(source, Mapping):
        source = _core.PhaseDiagramPropertyGrid.from_dict(dict(source))
    x, y, batches = _saved_property_grid_batches(source)
    shape = (len(y), len(x))
    values, fields = np.full(shape, np.nan), np.zeros(shape, int)
    equilibrium_success, covered = np.zeros(shape, bool), np.zeros(shape, bool)
    context, evaluate = None, None
    diagnostics, equilibrium_diagnostics = [], []
    _, options = _phase_selection(phases, include_phases, exclude_phases)
    if phase is not None:
        options["phase"] = phase
    if instance is not None:
        options["instance"] = instance
    if relaxed:
        options["relaxed"] = bool(relaxed)
    if averaging_scheme != _core.AveragingType.VRH:
        options["averaging_scheme"] = averaging_scheme.name
    for rows, cols, batch in batches:
        if not batch.has_solutions:
            raise ValueError(
                "The input grid must retain equilibria (store_solutions=True)."
            )
        current = batch.context.to_dict()
        copies = current["settings"]["max_phase_instances"]
        if current["phase_names"][::copies] != [p.name for p in phases]:
            raise ValueError(
                "Candidate models must match the saved grid's original order and names."
            )
        if context is None:
            evaluate, branch_context, _ = _property_evaluator(
                phases,
                property_name,
                phase=phase,
                instance=instance,
                include_phases=options.get("include_phases"),
                exclude_phases=options.get("exclude_phases", ()),
                diagram=diagram,
                relaxed=relaxed,
                averaging_scheme=averaging_scheme.name,
                copies=copies,
            )
            if branch_context is not None:
                _check_context(current, branch_context)
            context = current
            if instance is not None:
                options["instance"] = int(instance)
        elif current != context:
            raise ValueError("Grid chunks contain different diagram contexts.")
        if not np.array_equal(batch.x, x[cols]) or not np.array_equal(batch.y, y[rows]):
            raise ValueError("Chunk coordinates disagree with the complete grid axes.")
        fields[rows, cols], equilibrium_success[rows, cols] = (
            batch.field_numbers,
            batch.equilibrium_success,
        )
        covered[rows, cols] = True
        equilibrium_diagnostics.extend(batch.equilibrium_diagnostics)
        block = values[rows, cols]
        for j, i in np.argwhere(batch.equilibrium_success):
            solution = batch.solution_at(int(j), int(i))
            try:
                value = evaluate(
                    solution.state,
                    field=int(batch.field_numbers[j, i]),
                    position=(batch.x[i], batch.y[j]),
                    bases=solution.phase_bases,
                    compositions=solution.phase_compositions,
                )
                block[j, i] = value
            except (ValueError, RuntimeError) as error:
                diagnostics.append(
                    f"grid [{(rows.start or 0)+j}, {(cols.start or 0)+i}]: {property_name}: {error}"
                )
    if context is None or not covered.all():
        raise ValueError("Saved equilibrium batches do not cover the complete grid.")
    success = np.isfinite(values)
    quantity = "property: " + property_name
    if options:
        quantity += "; options=" + json.dumps(options)
    equilibrium_diagnostics = list(dict.fromkeys(equilibrium_diagnostics))
    return _core.PhaseDiagramPropertyGrid.from_dict(
        dict(
            format="burnman_cpp.property_grid",
            schema_version=1,
            context=context,
            x=x,
            y=y,
            relaxed=bool(relaxed),
            resolved=bool(
                not diagnostics and not np.any(~equilibrium_success & (fields > 0))
            ),
            values={quantity: values},
            field_numbers=fields,
            success=success,
            equilibrium_success=equilibrium_success,
            diagnostics=[*equilibrium_diagnostics, *diagnostics],
            equilibrium_diagnostics=equilibrium_diagnostics,
            equilibrium_solves=0,
            solutions=None,
        )
    )


def _property_boundary_values(data, phases, quantity):
    """Reuse saved boundary equilibria with the grid's property settings."""
    from .phase_diagram import _section_axes
    from .phase_diagram_labels import _field_observations

    name, _, encoded = quantity.removeprefix("property: ").partition("; options=")
    options = json.loads(encoded) if encoded else {}
    copies = data.get("settings", {}).get("max_phase_instances", 3)
    if data["phase_names"][::copies] != [p.name for p in phases]:
        raise ValueError(
            "Candidate models must match the diagram's original order and names."
        )
    evaluate, _, candidate = _property_evaluator(
        phases, name, diagram=data, copies=copies, **options
    )
    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    fields = {f["id"]: f for f in data.get("fields", [])}
    boundaries = {b["id"]: b for b in data.get("boundaries", [])}

    def sample(polygon):
        points, values = [], []
        limiting = None
        if candidate is not None:
            count = sum(i // copies == candidate for i in polygon.phases)
            if not count:
                return np.empty((0, 2)), np.empty(0)
            limiting = {
                key: {p[0] for p in states}
                for key, _, states in _field_observations(
                    polygon,
                    candidate,
                    count,
                    copies,
                    data.get("samples", []),
                    fields,
                    boundaries,
                    axes,
                    ranges,
                )
            }
        for line_id in polygon.boundary_ids:
            for index, state in enumerate(boundaries[line_id]["points"]):
                if not state.get("phases"):
                    continue
                if limiting is not None:
                    ids = limiting.get(("boundary", line_id, index))
                    if ids is None:
                        continue
                    state = dict(
                        state, phases=[p for p in state["phases"] if p["id"] in ids]
                    )
                position = tuple(state[a] for a in axes)
                try:
                    value = evaluate(
                        state,
                        field=polygon.field_number,
                        position=position,
                    )
                except (ValueError, RuntimeError):
                    value = np.nan
                points.append(position)
                values.append(value)
        return np.asarray(points).reshape(-1, 2), np.asarray(values)

    return sample
