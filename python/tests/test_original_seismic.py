# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_seismic.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
import burnman_cpp as burnman


class test_seismic(BurnManTest):

    def test_internal_depth_list(self):
        models = [
            burnman.seismic.PREM(),
            burnman.seismic.STW105(),
            burnman.seismic.AK135(),
            burnman.seismic.IASP91(),
        ]
        ref_depth_lists = {
            "PREM": [0.0, 6371000.0, 94],
            "STW105": [0.0, 6371000.0, 750],
            "AK135": [0.0, 6371000.0, 145],
            "IASP91": [0.0, 6371000.0, 152],
        }
        for model in models:
            dl = model.internal_depth_list()
            name = model.__class__.__name__
            stats = [min(dl), max(dl), len(dl)]
            self.assertArraysAlmostEqual(stats, ref_depth_lists[name])

    def test_evaluate(self):
        models = [
            burnman.seismic.PREM(),
            burnman.seismic.Fast(),
            burnman.seismic.Slow(),
            burnman.seismic.STW105(),
            burnman.seismic.AK135(),
            burnman.seismic.IASP91(),
        ]
        ref = {
            "PREM": [12817.6924, 6932.8549, 10010.358087588364, 5120.629099999999],
            "Fast": [
                12795.360742611414,
                6941.422556820191,
                9973.805377944691,
                5120.629099999999,
            ],
            "Slow": [
                12795.360742611414,
                6904.329188001315,
                10008.07546360332,
                5120.629099999999,
            ],
            "STW105": [
                12817.85012987013,
                6932.934206525182,
                10010.486818122406,
                5120.674925562243,
            ],
            "AK135": [
                12798.468686868688,
                6920.521212121212,
                9997.132035348877,
                5103.89696969697,
            ],
            "IASP91": [12794.4, 6921.0, 9991.48053893916],
        }
        for model in models:
            dl = model.internal_depth_list()
            name = model.__class__.__name__
            depth = 2000000.0
            vars = ["v_p", "v_s", "v_phi", "density"]
            if name == "IASP91":
                vars = vars[0:-1]
            result = model.evaluate(vars, [depth])
            result = list(result.T[0])
            self.assertArraysAlmostEqual(result, ref[name])
