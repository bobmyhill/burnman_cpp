"""High-level constraint construction and contour batches; solves use C++."""

from numbers import Integral

from .. import _core


def pseudosection_contour_levels(
    result,
    phases,
    levels,
    constraint,
    *,
    settings=None,
    composition=None,
    verbose=False,
):
    """Trace several constraint values through a completed pseudosection.

    ``result`` is a native result or its saved JSON dictionary. Supply the
    original candidate phases, in their original order. ``constraint(value)``
    returns any native equality constraint or a field-specific constraint
    factory. Levels use the units expected by that constructor (normally SI).
    ``settings`` controls native seeding and continuation; ``composition``
    supplies the elemental bulk for legacy results without composition_start.

    Return JSON-ready records with ``value`` and ``contours`` keys, retaining
    every level's resolution status and diagnostics. The saved diagram is
    unchanged. ``verbose=True`` prints progress after each level.
    """
    import numpy as np

    values = [float(value) for value in levels]
    if not np.isfinite(values).all():
        raise ValueError("Contour levels must be finite.")
    if not callable(constraint):
        raise TypeError("constraint must construct a constraint from a contour value.")
    if settings is None:
        settings = _core.PseudosectionContourSettings()
    phases = list(phases)
    records = []
    for value in values:
        lines = _core.pseudosection_contours(
            result,
            phases,
            constraint(value),
            settings=settings,
            composition=composition,
        )
        records.append(dict(value=value, contours=lines.to_dict()))
        if verbose:
            print(
                f"Contour {value:g}: {len(lines.lines)} segments; "
                f"{lines.equilibrium_solves} native solves; resolved={lines.resolved}",
                flush=True,
            )
            for diagnostic in lines.diagnostics:
                print(diagnostic, flush=True)
    return records


def _phase_composition_constraint_for_diagram(
    result, phase_name, site_names, numerator, denominator, value, *, instance=None
):
    """Constrain a named solution's composition across pseudosection fields.

    ``result`` is the saved diagram or its native result. ``numerator`` and
    ``denominator`` give weights for ``site_names``; site multiplicities are
    included by the native PhaseCompositionConstraint. For Mg/(Mg+Fe2+), use
    the model's Mg and ferrous-Fe sites with weights [1, 0] and [1, 1].

    Return a factory accepted by the native contour tracer. Fields without
    the phase, active pure faces, or faces lacking the requested sites are
    skipped. Multiple coexisting copies require a one-based ``instance`` or
    the exact saved name, e.g. 'g #2'. Instance numbers identify saved branches;
    they do not rank compositions by Mg or Fe content.
    """
    names = (
        result.get("phase_names", [])
        if isinstance(result, dict)
        else result.phase_names
    )
    if instance is not None:
        if (
            isinstance(instance, bool)
            or not isinstance(instance, Integral)
            or instance < 1
        ):
            raise ValueError("instance must be a positive integer, starting at 1.")
        if " #" in phase_name:
            raise ValueError(
                "Select an instance by its saved name or by instance, not both."
            )
        phase_name += f" #{instance}" if instance > 1 else ""
    exact = instance is not None or " #" in phase_name
    selected = {
        i
        for i, name in enumerate(names)
        if (name if exact else name.split(" #")[0]) == phase_name
    }
    if not selected:
        raise ValueError(f"No saved candidate phase named {phase_name!r}.")
    sites = tuple(site_names)

    def factory(assemblage, parameters, phase_ids):
        indices = [i for i, phase_id in enumerate(phase_ids) if phase_id in selected]
        if not indices:
            return None
        if len(indices) > 1:
            raise ValueError(f"Multiple copies of {phase_name!r}; select an instance.")
        index = indices[0]
        phase = assemblage.get_phase(index)
        if not isinstance(phase, _core.Solution) or not all(
            site in phase.site_names for site in sites
        ):
            return None
        return _core.PhaseCompositionConstraint(
            index, sites, numerator, denominator, value, assemblage, parameters
        )

    return factory


_core.PhaseCompositionConstraint.for_diagram = staticmethod(
    _phase_composition_constraint_for_diagram
)
