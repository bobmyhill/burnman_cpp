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

"""Display options for contours interpolated from saved grids."""

import json

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params


@pytest.fixture
def section():
    phase = bm.Mineral(oxide_params())
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    settings.max_phase_instances = 1
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], (3.0e9, 5.0e9), (800.0, 1200.0), settings
    )
    return result, phase


@pytest.mark.parametrize(
    "variable,value,expected_si,expected_display",
    [
        ("P", 4.0e9, "4e+09", "40"),
        ("T", 1000, "1000", "726.85"),
        (None, 0.7, "0.7", "0.7"),
    ],
)
@pytest.mark.parametrize(
    "display", [{}, dict(pressure_unit="kbar", temperature_unit="C")]
)
def test_contour_labels_inherit_units_and_allow_explicit_display_options(
    section, variable, value, expected_si, expected_display, display
):
    import matplotlib.pyplot as plt

    expected = expected_display if display else expected_si
    result, phase = section
    equilibria = bm.phase_diagram_grid(
        result, [phase], "temperature", resolution=(3, 3), store_solutions=True
    )
    scalar = bm.phase_diagram_property_grid(equilibria, [phase], "temperature")
    contours = bm.phase_diagram_grid_contours(
        scalar, [1000.0], diagram=result, phases=[phase]
    )[0]["contours"]
    fig, ax = bm.plot_phase_diagram(result, label_key_path=None, **display)
    bm.plot_phase_diagram_contours(
        result,
        contours,
        ax,
        variable=variable,
        value=value,
        label_fontsize=8.0,
        label_placement="single",
        **display,
    )
    assert len([t for t in ax.texts if t.get_text() == expected]) == 1
    assert ax.collections[-1].get_label() == expected
    plt.close(fig)


@pytest.mark.parametrize(
    "placement,expected_y", [("single", 0.2), ("coexistence", 0.8)]
)
def test_label_placement_can_prefer_coexistence_over_the_longest_segment(
    placement, expected_y
):
    import matplotlib.pyplot as plt

    saved = dict(diagram_type="PT", pressure_range=(0, 1), temperature_range=(0, 1))
    contours = dict(
        diagram_type="PT",
        lines=[
            dict(
                phases=[0],
                points=[
                    dict(temperature=0.05, pressure=0.2),
                    dict(temperature=0.95, pressure=0.2),
                ],
            ),
            dict(
                phases=[0, 1, 2],
                points=[
                    dict(temperature=0.2, pressure=0.8),
                    dict(temperature=0.8, pressure=0.8),
                ],
            ),
        ],
    )
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    bm.plot_phase_diagram_contours(
        saved,
        contours,
        ax,
        value=0.5,
        label_format=".2f",
        label_placement=placement,
        pressure_unit="Pa",
        temperature_unit="K",
    )
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "0.50"
    assert ax.texts[0].get_position()[1] == pytest.approx(expected_y)
    plt.close(fig)


@pytest.mark.parametrize(
    "units", [dict(), dict(volume_unit="kg/m3", entropy_unit="kB/atom", swap_axes=True)]
)
def test_separate_contour_plotter_inherits_units_and_accepts_json(units):
    import matplotlib.pyplot as plt

    phase = bm.Mineral(oxide_params())
    phase.set_state(4.0e9, 1000.0)
    v, s = phase.molar_volume, phase.molar_entropy
    options = bm.PhaseDiagramSettings()
    options.max_phase_instances = 1
    options.entropy_seeds = options.volume_seeds = 2
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        diagram="SV",
        entropy_range=(s - 0.05, s + 0.05),
        volume_range=(v - 1.0e-9, v + 1.0e-9),
        pressure_range=(0.0, 8.0e9),
        temperature_range=(500.0, 1500.0),
        settings=options,
    )
    equilibria = bm.phase_diagram_grid(
        result, [phase], "volume", resolution=(3, 3), store_solutions=True
    )
    scalar = bm.phase_diagram_property_grid(equilibria, [phase], "volume")
    contours = bm.phase_diagram_grid_contours(
        scalar, [v], diagram=result, phases=[phase]
    )[0]["contours"]
    fig, ax = bm.plot_phase_diagram(result, label_key_path=None, **units)
    collections = len(ax.collections)
    bm.plot_phase_diagram_contours(
        result,
        json.loads(json.dumps(contours)),
        ax,
        label="constant V",
        label_fontsize=9.0,
    )
    assert len(ax.collections) == collections + 1
    points = ax.collections[-1].get_paths()[0].vertices
    if units:
        expected_density = (
            sum(
                bm.Composition(
                    result.composition_start, "molar"
                ).mass_composition.values()
            )
            / v
        )
        np.testing.assert_allclose(points[:, 1], expected_density, rtol=1.0e-9)
        assert all(
            text.get_fontsize() == 9.0
            for text in ax.texts
            if text.get_text() == "constant V"
        )
    else:
        np.testing.assert_allclose(points[:, 0], v, rtol=1.0e-9)
    with pytest.raises(ValueError, match="must match"):
        bm.plot_phase_diagram_contours(result, contours, ax, temperature_unit="C")
    plt.close(fig)
