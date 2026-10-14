# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Single-point phase discovery uses the same extensive constraints as a grid."""

import pytest
import burnman_cpp as bm
from conftest import oxide_params


@pytest.mark.parametrize("pair", ["PT", "PS", "PV", "TS", "TV", "SV"])
@pytest.mark.parametrize("amount", [1.0, 2.0])
def test_single_point_constraints_preserve_bulk_scale_and_discover_phases(pair, amount):
    stable = bm.Mineral(oxide_params())
    unstable_params = oxide_params(energy_shift=100000.0)
    unstable_params["name"] = "high energy"
    unstable = bm.Mineral(unstable_params)
    bulk = {"Mg": amount, "O": amount}
    stable.set_state(4.0e9, 1000.0)
    targets = dict(
        P=4.0e9,
        T=1000.0,
        S=amount * stable.molar_entropy,
        V=amount * stable.molar_volume,
    )
    names = dict(P="pressure", T="temperature", S="entropy", V="volume")
    options = {names[axis]: targets[axis] for axis in pair}
    state = bm.stable_equilibrium(
        bulk,
        [unstable, stable],
        **options,
        pressure_range=(2.0e9, 5.0e9),
        temperature_range=(500.0, 1300.0),
    )
    assert state.success, state.message
    assert len(state.phases) == 1
    assert state.phases[0].candidate_index == 1
    assert state.phases[0].amount == pytest.approx(amount, rel=1.0e-9)
    for axis, target in targets.items():
        assert getattr(state, names[axis]) == pytest.approx(target, rel=1.0e-7)


@pytest.mark.parametrize(
    "fixed",
    [
        dict(pressure=1.0e9),
        dict(pressure=1.0e9, temperature=1000.0, entropy=10.0),
        dict(pressure=-1.0, entropy=10.0),
        dict(temperature=1000.0, volume=0.0),
        dict(pressure=1.0e9, entropy=float("nan")),
    ],
)
def test_single_point_rejects_invalid_constraints(fixed):
    with pytest.raises(ValueError):
        bm.stable_equilibrium(
            {"Mg": 1.0, "O": 1.0},
            [bm.Mineral(oxide_params())],
            **fixed,
            pressure_range=(0.0, 2.0e9),
        )
