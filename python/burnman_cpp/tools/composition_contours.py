# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Arithmetic composition expressions and contours from retained equilibria."""

import ast
import json
from collections.abc import Mapping
from numbers import Integral
import operator

import numpy as np

from .. import _core
from .properties import _saved_property_grid_batches, load_property_grid

_VOLUME_SYMBOLS = {"volume_fraction", "standard_volume_fraction"}


def _diagram_context(data):
    return _core.PhaseDiagramResult.from_dict(
        dict(data, boundaries=[], samples=[], nodes=[], fields=[])
    ).to_dict()


def _check_context(saved, diagram):
    keys = (
        "diagram_type",
        "phase_names",
        "composition_start",
        "composition_end",
        "pressure_range",
        "temperature_range",
        "entropy_range",
        "volume_range",
        "composition_range",
        "fixed_coordinate",
        "fixed_value",
    )
    if any(saved.get(key) != diagram.get(key) for key in keys) or (
        saved["settings"]["max_phase_instances"]
        != diagram["settings"]["max_phase_instances"]
    ):
        raise ValueError(
            "The diagram does not match the stored grid's axes, bulk or candidate models."
        )


def _arithmetic_expression(text, available, *, kind="composition"):
    """Interpret arithmetic only; never execute user text as Python code."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("expression must be a nonempty arithmetic expression.")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid {kind} expression: {error.msg}") from error
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        *operations,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(
                "Expressions allow names, numbers, parentheses and +, -, *, / only."
            )
        if isinstance(node, ast.Constant) and (
            type(node.value) not in (int, float) or not np.isfinite(float(node.value))
        ):
            raise ValueError("Expression constants must be finite numbers.")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    available = set(available)
    if names - available:
        raise ValueError(
            f"Unknown {kind} symbols: {', '.join(sorted(names - available))}. "
            f"Available: {', '.join(sorted(available))}"
        )

    def evaluate(node, values):
        if isinstance(node, ast.Constant):
            return np.float64(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            return operations[type(node.op)](evaluate(node.operand, values))
        return operations[type(node.op)](
            evaluate(node.left, values), evaluate(node.right, values)
        )

    return lambda values: evaluate(tree.body, values), names


def _expression(text, symbols):
    """Bind the shared arithmetic interpreter to composition coefficients."""
    evaluate, names = _arithmetic_expression(text, symbols.keys() | _VOLUME_SYMBOLS)

    def apply(coefficients, extra=None):
        values = {name: coefficients @ symbols[name] for name in names & symbols.keys()}
        values.update(extra or {})
        return evaluate(values)

    return apply, names


def _volume_fractions(phases, exclude_phases=()):
    """Replay volumes on private copies, retaining amounts and compositions."""
    from .phase_diagram import _value

    materials = [phase.copy() for phase in phases]
    excluded = {i for i, phase in enumerate(phases) if phase.name in exclude_phases}

    def fraction(state, selected_ids, *, standard=False):
        pressure, temperature = (
            (1.0e5, 298.15)
            if standard
            else (_value(state, "pressure"), _value(state, "temperature"))
        )

        def volume(phase):
            amount = _value(phase, "amount")
            if amount <= 0.0:
                return 0.0
            material = materials[_value(phase, "candidate_index")]
            if isinstance(material, _core.Solution):
                material.set_composition(_value(phase, "composition"))
            if (
                not material.has_state()
                or material.pressure != pressure
                or material.temperature != temperature
            ):
                material.set_state(pressure, temperature)
            result = amount * material.molar_volume
            if not np.isfinite(result) or result <= 0.0:
                raise ValueError(
                    "A phase has no positive finite volume at the requested state."
                )
            return result

        states = [
            p
            for p in _value(state, "phases")
            if _value(p, "candidate_index") not in excluded
        ]
        if not any(_value(p, "amount") > 0.0 for p in states):
            raise ValueError(
                "No positive phase amounts remain in the volume denominator."
            )
        if not any(
            _value(p, "id") in selected_ids and _value(p, "amount") > 0.0
            for p in states
        ):
            return 0.0
        total = _value(state, "volume", 0.0) if not standard and not excluded else 0.0
        if not np.isfinite(total) or total <= 0.0:
            contributions = [(p, volume(p)) for p in states]
            total = sum(v for _, v in contributions)
            selected = sum(
                v for p, v in contributions if _value(p, "id") in selected_ids
            )
        else:
            selected = sum(volume(p) for p in states if _value(p, "id") in selected_ids)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                "The saved assemblage has no positive finite total volume."
            )
        result = selected / total
        if not 0.0 <= result <= 1.0 + 1.0e-8:
            raise ValueError(
                "Phase volumes are inconsistent with the saved total volume."
            )
        return min(result, 1.0)

    return fraction


def _composition_symbols(phase):
    """Linear weights on original endmembers, independent of reduced faces."""
    if not isinstance(phase, _core.Solution):
        return {
            element: np.array([amount]) for element, amount in phase.formula.items()
        }
    model = phase.solution_model
    symbols = dict(zip(model.site_names, np.asarray(model.endmember_occupancies).T))
    formulae = phase.endmember_formulae
    for element in set().union(*formulae):
        symbols[element] = np.array([formula.get(element, 0.0) for formula in formulae])
    for name, weights in zip(phase.endmember_names, np.eye(phase.n_endmembers)):
        if name.isidentifier():
            symbols[f"X_{name}"] = weights
    return symbols


def phase_composition_grid(
    source, phases, phase, expression, *, instance=None, diagram=None, exclude_phases=()
):
    """Evaluate a composition expression on a stored equilibrium grid.

    source accepts a retained property grid, its JSON dictionary/file, or a
    completed chunk manifest/directory. Supply the original candidate phases
    in their original order. No equilibrium routines run, and candidate models
    are unchanged. Original endmember coefficients handle reduced active faces.

    Symbols are the printed site names (fractional occupancies), elemental atom
    counts per formula unit, and X_<endmember> coefficients. For garnet, use
    ``Mgx_A/(Mgx_A+Fex_A)`` to exclude ferric Fe, or ``Mg/(Mg+Fe)`` for total Fe.
    Numbers, parentheses, unary signs and +, -, *, / are supported.
    ``volume_fraction`` is n_phase*V_phase/V_total at the stored P,T;
    ``standard_volume_fraction`` evaluates every phase volume at 1 bar,
    298.15 K, freezing saved amounts and compositions. Both use private model
    copies. Absent phases have volume fraction zero; unselected coexisting
    copies are summed for expressions using only these volume symbols.
    ``exclude_phases`` omits named candidate models (all their copies) from
    the volume denominator, e.g. ("H2O",) for a fluid-free normalization.
    The contoured phase must remain included. Exclusions are saved in the
    quantity key and automatically reused for boundary values.

    phase selects a candidate name. Chemical expressions for coexisting copies
    require a one-based instance or an exact name such as 'fsp #2'. These are
    local solver copies.
    Alternatively select a secondary label such as 'fsp_1' and supply its
    completed diagram; compositions then identify the branch at each node.

    Return a PhaseDiagramPropertyGrid with one values entry named
    '<phase>: <expression>' (plus any volume exclusions) and equilibrium_solves=0.
    Missing phases are NaN for chemical expressions and zero for volume-only
    expressions. Failed equilibria and undefined arithmetic are NaN, with
    success=False. Original
    equilibrium_success is preserved, including unrelated property failures.
    Save with save_property_grid(); extract lines with phase_composition_contours().
    The returned scalar grid does not duplicate the stored equilibrium states.
    """
    phases = list(phases)
    exclude_phases = (
        (exclude_phases,) if isinstance(exclude_phases, str) else tuple(exclude_phases)
    )
    if any(name not in {p.name for p in phases} for name in exclude_phases):
        raise ValueError("Excluded phases must use the original candidate model names.")
    if not isinstance(phase, str):
        raise TypeError("phase must be a candidate name or secondary label.")
    if instance is not None and (
        isinstance(instance, bool) or not isinstance(instance, Integral) or instance < 1
    ):
        raise ValueError("instance must be a positive integer, starting at 1.")
    name, separator, copy = phase.partition(" #")
    if separator:
        if instance is not None or not copy.isdecimal() or int(copy) < 1:
            raise ValueError("Select a valid saved copy by name or instance, not both.")
        instance = int(copy)
    quantity_phase = (
        f"{name} #{instance}" if instance is not None and instance > 1 else phase
    )
    candidates = [i for i, model in enumerate(phases) if model.name == name]
    branch, branch_context = None, None
    if not candidates and diagram is not None and instance is None:
        candidate, branch, branch_context = _secondary_branch(diagram, phase)
    elif len(candidates) == 1:
        candidate = candidates[0]
    else:
        raise ValueError(
            f"No unique candidate named {phase!r}; secondary labels require diagram=."
        )
    if not 0 <= candidate < len(phases):
        raise ValueError(
            "The secondary-labelled diagram uses different candidate models."
        )
    model = phases[candidate]
    evaluate, symbols = _expression(expression, _composition_symbols(model))
    volume_symbols = symbols & _VOLUME_SYMBOLS
    volume_only = bool(volume_symbols) and symbols <= _VOLUME_SYMBOLS
    if exclude_phases and (not volume_symbols or model.name in exclude_phases):
        raise ValueError(
            "Volume exclusions require a volume expression and must retain the contoured phase."
        )
    volume = _volume_fractions(phases, exclude_phases) if volume_symbols else None
    quantity = f"{quantity_phase}: {expression}"
    if exclude_phases:
        quantity += "; exclude_phases=" + json.dumps(exclude_phases)
    n_endmembers = model.n_endmembers if isinstance(model, _core.Solution) else 1
    if isinstance(source, Mapping):
        source = _core.PhaseDiagramPropertyGrid.from_dict(dict(source))
    x, y, batches = _saved_property_grid_batches(source)
    shape = (len(y), len(x))
    coefficients = np.full((*shape, n_endmembers), np.nan)
    extra = {name: np.full(shape, np.nan) for name in volume_symbols}
    field_numbers = np.zeros(shape, int)
    equilibrium_success = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    equilibrium_diagnostics, evaluation_diagnostics, context = [], [], None
    for rows, cols, batch in batches:
        if not batch.has_solutions:
            raise ValueError(
                "The input grid must retain equilibria (store_solutions=True)."
            )
        current = batch.context.to_dict()
        copies = current["settings"]["max_phase_instances"]
        if instance is not None and instance > copies:
            raise ValueError(f"instance must be between 1 and {copies} for this grid.")
        if current["phase_names"][::copies] != [p.name for p in phases]:
            raise ValueError(
                "Candidate models must match the saved grid's original order and names."
            )
        if context is not None and current != context:
            raise ValueError("Grid chunks contain different diagram contexts.")
        if branch_context is not None:
            _check_context(current, branch_context)
        if not np.array_equal(batch.x, x[cols]) or not np.array_equal(batch.y, y[rows]):
            raise ValueError("Chunk coordinates disagree with the complete grid axes.")
        context = current
        equilibrium_diagnostics.extend(batch.equilibrium_diagnostics)
        field_numbers[rows, cols] = batch.field_numbers
        equilibrium_success[rows, cols] = batch.equilibrium_success
        covered[rows, cols] = True
        block = coefficients[rows, cols]
        for j, i in np.argwhere(batch.equilibrium_success):
            state = batch.solution_at(int(j), int(i)).state
            minimum = current["settings"]["amount_tolerance"] * sum(
                p.amount for p in state.phases
            )
            if volume_only:
                minimum = 0.0
            matches = [
                p
                for p in state.phases
                if p.candidate_index == candidate and p.amount > minimum
            ]
            if branch is not None:
                position = (x[cols][i], y[rows][j])
                matches = branch(matches, int(batch.field_numbers[j, i]), position)
            elif instance is not None:
                matches = [p for p in matches if p.id % copies == instance - 1]
            elif len(matches) > 1 and not volume_only:
                raise ValueError(
                    f"{phase!r} has coexisting copies; select instance= or a secondary label."
                )
            if matches:
                if len(matches[0].composition) != n_endmembers:
                    raise ValueError(
                        "Saved composition does not match the original candidate endmembers."
                    )
                block[j, i] = matches[0].composition
            if volume_symbols:
                try:
                    for name in volume_symbols:
                        extra[name][rows, cols][j, i] = volume(
                            state,
                            {p.id for p in matches},
                            standard=name == "standard_volume_fraction",
                        )
                except (ValueError, RuntimeError) as error:
                    evaluation_diagnostics.append(
                        f"grid [{(rows.start or 0)+j}, {(cols.start or 0)+i}]: {error}"
                    )
    if context is None or not covered.all():
        raise ValueError("Saved equilibrium batches do not cover the complete grid.")
    present = (
        equilibrium_success if volume_only else np.isfinite(coefficients).all(axis=-1)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = np.broadcast_to(evaluate(coefficients, extra), shape).copy()
    success = present & np.isfinite(values)
    undefined = int(np.count_nonzero(present & ~success))
    values[~success] = np.nan
    equilibrium_diagnostics = list(dict.fromkeys(equilibrium_diagnostics))
    diagnostics = [*equilibrium_diagnostics, *evaluation_diagnostics]
    if undefined:
        diagnostics.append(
            f"{undefined} composition values are undefined or nonfinite; masked."
        )
    return _core.PhaseDiagramPropertyGrid.from_dict(
        dict(
            format="burnman_cpp.property_grid",
            schema_version=1,
            context=context,
            x=x,
            y=y,
            relaxed=False,
            resolved=bool(
                not undefined and not np.any(~equilibrium_success & (field_numbers > 0))
            ),
            values={quantity: values},
            field_numbers=field_numbers,
            success=success,
            equilibrium_success=equilibrium_success,
            diagnostics=list(dict.fromkeys(diagnostics)),
            equilibrium_diagnostics=equilibrium_diagnostics,
            equilibrium_solves=0,
            solutions=None,
        )
    )


def phase_composition_contours(grid, levels, *, diagram, phases=None):
    """Interpolate field-clipped contours from a saved scalar grid; never solve.

    grid is phase_composition_grid()'s or phase_diagram_property_grid()'s output,
    or its saved JSON path. Also exported as phase_diagram_grid_contours(). Return
    JSON-ready {value, contours} records, usable with plot_phase_diagram_contours().
    Supply the matching completed diagram for exact field edges and holes.
    If phases is supplied, also evaluate the expression/property on pre-existing
    boundary equilibria. Missing boundary vertices use a local affine estimate
    from that field's observations. No new boundary or equilibrium solves run.
    A separate interpolant is built in each field and clipped to its polygon;
    contours reach field edges but never continue outside them. Invalid grid
    nodes mask adjacent triangles. Accuracy depends on the retained sampling.
    Requires the Matplotlib and Shapely examples dependencies.
    """
    from ._grid_contours import field_contours
    from .phase_diagram import _phase_diagram_data

    if not isinstance(grid, _core.PhaseDiagramPropertyGrid):
        grid = load_property_grid(grid)
    if len(grid.values) != 1 or min(len(grid.x), len(grid.y)) < 2:
        raise ValueError(
            "Contours need one scalar property and at least two points per axis."
        )
    levels = np.asarray(list(levels), dtype=float)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all():
        raise ValueError("levels must contain finite contour values.")
    if len(np.unique(levels)) != len(levels):
        raise ValueError("Contour levels must be distinct.")
    quantity, values = next(iter(grid.values.items()))
    data = _phase_diagram_data(diagram)
    _check_context(grid.context.to_dict(), _diagram_context(data))
    geometry = _core.phase_diagram_field_polygons(data)
    boundary_values = (
        _boundary_expression(data, list(phases), quantity)
        if phases is not None
        else None
    )
    records = [
        dict(
            value=float(level),
            contours=dict(
                schema_version=1,
                diagram_type=grid.diagram_type,
                quantity=quantity,
                method="saved_grid_and_boundary_interpolation",
                lines=[],
                diagnostics=grid.diagnostics,
                equilibrium_solves=0,
                resolved=grid.resolved,
            ),
        )
        for level in levels
    ]
    names = dict(
        P="pressure",
        T="temperature",
        S="entropy",
        V="volume",
        X="composition_coordinate",
    )
    axes = [names[c] for c in grid.diagram_type]
    coordinates = np.stack(np.meshgrid(grid.x, grid.y), axis=-1)
    by_level = {record["value"]: record["contours"] for record in records}
    for polygon in geometry.polygons:
        if polygon.outside_model_domain:
            continue
        field = grid.field_numbers == polygon.field_number
        valid = field & grid.success & np.isfinite(values)
        points, samples = np.empty((0, 2)), np.empty(0)
        if boundary_values is not None:
            points, samples = boundary_values(polygon)
        boundary_count = int(np.isfinite(samples).sum())
        points = np.vstack([points, coordinates[valid]])
        samples = np.r_[samples, values[valid]]
        if len(points) < 3:
            continue
        for level, segment in field_contours(
            points, samples, polygon, coordinates[field & ~valid], levels
        ):
            by_level[level]["lines"].append(
                dict(
                    field_number=polygon.field_number,
                    phases=list(polygon.phases),
                    preexisting_boundary_samples=boundary_count,
                    closed=bool(np.array_equal(segment[0], segment[-1])),
                    termination="Interpolated and clipped to the saved field",
                    points=[dict(zip(axes, map(float, point))) for point in segment],
                )
            )
    return records


def _boundary_expression(data, phases, quantity):
    """Evaluate saved limiting compositions on the appropriate field side."""
    if quantity.startswith("property: "):
        from .property_contours import _property_boundary_values

        return _property_boundary_values(data, phases, quantity)
    from .phase_diagram import _section_axes
    from ._boundary_states import field_boundary_states
    from .phase_diagram_labels import (
        label_phase_diagram_solutions,
        _field_observations,
        _track_compositions,
    )

    quantity, _, exclusions = quantity.partition("; exclude_phases=")
    exclude_phases = json.loads(exclusions) if exclusions else ()
    phase, expression = quantity.rsplit(": ", 1)
    name, separator, copy = phase.partition(" #")
    instance = int(copy) if separator else 1
    candidates = [i for i, p in enumerate(phases) if p.name == name]
    labels = None
    if candidates:
        candidate = candidates[0]
    else:
        annotations = label_phase_diagram_solutions(data)["secondary_phase_labels"]
        group = next((g for g in annotations["groups"] if g["label"] == phase), None)
        if group is None:
            raise ValueError(f"Unknown secondary phase label {phase!r}.")
        candidate = group["candidate_index"]
        labels = {
            f["field_number"]: {p["phase_id"]: p["label"] for p in f["phases"]}
            for f in annotations["fields"]
        }
    evaluate, symbols = _expression(expression, _composition_symbols(phases[candidate]))
    volume_symbols = symbols & _VOLUME_SYMBOLS
    volume_only = bool(volume_symbols) and symbols <= _VOLUME_SYMBOLS
    volume = _volume_fractions(phases, exclude_phases) if volume_symbols else None
    copies = data.get("settings", {}).get("max_phase_instances", 3)
    if data["phase_names"][::copies] != [p.name for p in phases]:
        raise ValueError(
            "Candidate models must match the diagram's original order and names."
        )
    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    fields = {f["id"]: f for f in data.get("fields", [])}
    boundaries = {b["id"]: b for b in data.get("boundaries", [])}
    boundary_states = field_boundary_states(data, phases) if volume_symbols else None

    def sample(polygon):
        ids = [i for i in polygon.phases if i // copies == candidate]
        empty = (np.empty((0, 2)), np.empty(0))
        if not ids or instance > len(ids):
            return empty
        if (
            labels is not None
            and phase not in labels.get(polygon.field_number, {}).values()
        ):
            return empty
        records = _field_observations(
            polygon,
            candidate,
            len(ids),
            copies,
            data.get("samples", []),
            fields,
            boundaries,
            axes,
            ranges,
        )
        if not records:
            return empty
        tracks = _track_compositions(records) if labels is not None else None
        limiting = (
            {key: state for key, _, state in boundary_states(polygon)}
            if boundary_states is not None
            else {}
        )
        slot = None
        if labels is not None:
            slot = next(
                i
                for i, phase_id in enumerate(ids)
                if labels[polygon.field_number][phase_id] == phase
            )
        points, compositions = [], []
        extra = {name: [] for name in volume_symbols}
        for index, (key, position, states) in enumerate(records):
            if key[0] != "boundary":
                continue
            if volume_symbols and key not in limiting:
                continue
            selected = (
                states[tracks[index].index(slot)]
                if tracks is not None
                else sorted(states, key=lambda s: s[0])[instance - 1]
            )
            points.append(position * np.diff(ranges).ravel() + ranges[:, 0])
            compositions.append(selected[1])
            if volume_symbols:
                selected_ids = (
                    {p[0] for p in states}
                    if volume_only and not separator and labels is None
                    else {selected[0]}
                )
                state = limiting[key]
                for name in volume_symbols:
                    try:
                        value = volume(
                            state,
                            selected_ids,
                            standard=name == "standard_volume_fraction",
                        )
                    except (ValueError, RuntimeError):
                        value = np.nan
                    extra[name].append(value)
        if not points:
            return empty
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = np.broadcast_to(
                evaluate(
                    np.asarray(compositions),
                    {k: np.asarray(v) for k, v in extra.items()},
                ),
                (len(points),),
            ).copy()
        return np.asarray(points), values

    return sample


def _secondary_branch(diagram, label):
    """Match each node to the nearest saved, labelled composition tracks."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from .phase_diagram import _phase_diagram_data, _section_axes, _value
    from .phase_diagram_labels import (
        label_phase_diagram_solutions,
        _field_observations,
        _track_compositions,
    )

    data = label_phase_diagram_solutions(_phase_diagram_data(diagram))
    annotations = data["secondary_phase_labels"]
    group = next((g for g in annotations["groups"] if g["label"] == label), None)
    if group is None:
        raise ValueError(f"Unknown secondary phase label {label!r}.")
    candidate = group["candidate_index"]
    copies = data["settings"]["max_phase_instances"]
    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    fields = {f["id"]: f for f in data["fields"]}
    boundaries = {b["id"]: b for b in data["boundaries"]}
    labels = {
        f["field_number"]: {p["phase_id"]: p["label"] for p in f["phases"]}
        for f in annotations["fields"]
    }
    references = {}
    for polygon in _core.phase_diagram_field_polygons(data).polygons:
        ids = [i for i in polygon.phases if i // copies == candidate]
        if label not in labels.get(polygon.field_number, {}).values():
            continue
        records = _field_observations(
            polygon,
            candidate,
            len(ids),
            copies,
            data["samples"],
            fields,
            boundaries,
            axes,
            ranges,
        )
        tracks = _track_compositions(records)
        compositions = [
            np.array([r[2][t.index(slot)][1] for slot in range(len(ids))])
            for r, t in zip(records, tracks)
        ]
        slot = next(
            i
            for i, phase_id in enumerate(ids)
            if labels[polygon.field_number][phase_id] == label
        )
        references[polygon.field_number] = (
            cKDTree([r[1] for r in records]),
            compositions,
            slot,
        )

    def select(phases, field, position):
        if field not in references or not phases:
            return []
        tree, compositions, slot = references[field]
        _, near = tree.query(
            (np.asarray(position) - ranges[:, 0]) / np.diff(ranges).ravel()
        )
        reference = compositions[near]
        if len(phases) != len(reference):
            raise ValueError(
                "Stored phase counts disagree with the secondary-labelled field."
            )
        rows, cols = linear_sum_assignment(
            cdist(reference, [_value(p, "composition") for p in phases])
        )
        return [phases[cols[np.flatnonzero(rows == slot)[0]]]]

    return candidate, select, _diagram_context(data)


# The same geometry and clipping apply to all scalar property grids.
phase_diagram_grid_contours = phase_composition_contours
