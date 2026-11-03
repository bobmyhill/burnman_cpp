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

"""A completed equilibrium boundary covers a failed retrace near zero kelvin."""

from copy import deepcopy

import pytest

import burnman_cpp as bm
from test_phase_diagram import curved_phases, settings


@pytest.mark.parametrize("short_tail", [False, True])
def test_uncovered_tail_is_not_discarded_as_a_partial_duplicate(short_tail):
    bulk = {"Mg": 1.0, "O": 1.0}
    phases = curved_phases()
    opts = settings()
    opts.max_recovery_passes = 0
    saved = bm.phase_diagram(
        bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), opts
    ).to_dict()
    complete = saved["boundaries"][0]
    whole = deepcopy(complete)
    whole["points"] = whole["points"][:-1]
    # The purported completed cover ends halfway along the accepted branch.
    # An open record with the same reaction contains a genuine uncovered tail.
    halfway = len(whole["points"]) - 2 if short_tail else len(whole["points"]) // 2
    end_node = deepcopy(saved["nodes"][complete["end_node"]])
    end_node["id"] = len(saved["nodes"])
    for name in ("pressure", "temperature", "entropy", "volume"):
        end_node[name] = complete["points"][halfway][name]
    saved["nodes"].append(end_node)
    complete["points"] = complete["points"][: halfway + 1]
    complete["end_node"] = end_node["id"]
    whole.update(id=1, end_node=-1, termination="domain edge; corrector failed")
    saved["boundaries"].append(whole)
    saved["resolved"] = False
    result = bm.refine_phase_diagram(bulk, phases, saved, opts)
    assert len(result.boundaries) >= 2
    assert any(b.end_node < 0 or b.start_node < 0 for b in result.boundaries)
    assert not result.resolved
