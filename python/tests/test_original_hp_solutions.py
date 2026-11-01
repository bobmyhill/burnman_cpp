# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_hp_solutions.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    native_endmembers,
    is_native_factory,
    ideal_model_call,
)
import burnman_cpp as burnman
from burnman_cpp import Mineral, CombinedMineral, Solution
from burnman_cpp.constants import gas_constant
from burnman_cpp.minerals import (
    HP_2011_fluids,
    HP_2011_ds62,
    HGP_2018_ds633,
    ig50NCKFMASHTOCr,
    ig50NCKFMASTOCr,
    mb50NCKFMASHTO,
    mp50KFMASH,
    mp50MnNCKFMASHTO,
    mp50NCKFMASHTO,
)
import numpy as np


def assert_linear_correction(test, mineral, raw, amounts, correction):
    """Check a dataset recipe through independent native thermodynamic values."""
    energy, entropy, volume = correction
    for P, T in [(1.0e5, 800.0), (1.0e9, 1200.0), (3.0e9, 1600.0)]:
        with test.subTest(P=P, T=T):
            mineral.set_state(P, T)
            for endmember in raw:
                endmember.set_state(P, T)
            for prop, expected, atol in (
                ("molar_gibbs", energy - T * entropy + P * volume, 1.0e-7),
                ("molar_enthalpy", energy + P * volume, 1.0e-7),
                ("molar_entropy", entropy, 1.0e-10),
                ("molar_volume", volume, 1.0e-18),
                ("molar_heat_capacity_p", 0.0, 1.0e-10),
            ):
                with test.subTest(property=prop):
                    base = sum(a * getattr(m, prop) for a, m in zip(amounts, raw))
                    test.assertFloatEqual(
                        getattr(mineral, prop) - base,
                        expected,
                        tol=1.0e-10,
                        tol_zero=atol,
                    )


def named_endmember(phase, name):
    return native_endmembers(phase)[phase.endmember_names.index(name)][0]


class hp_solutions(BurnManTest):

    def test_metabasite_melt_speciation_correction(self):
        anorthite = named_endmember(mb50NCKFMASHTO.L(), "anoL")
        self.assertEqual(anorthite.formula, {"Ca": 1, "Al": 2, "Si": 2, "O": 8})
        assert_linear_correction(
            self,
            anorthite,
            [HP_2011_ds62.woL(), HP_2011_ds62.silL()],
            [1.0, 1.0],
            [-46500.0, 0.0, -2.5e-6],
        )

    def test_staurolite_dqf_correction(self):
        for dataset in (mp50KFMASH, mp50MnNCKFMASHTO, mp50NCKFMASHTO):
            with self.subTest(dataset=dataset.__name__):
                for mineral in (dataset.mstm(), named_endmember(dataset.st(), "mstm")):
                    self.assertEqual(mineral.formula, HP_2011_ds62.mst().formula)
                    assert_linear_correction(
                        self, mineral, [HP_2011_ds62.mst()], [1.0], [-8000.0, 0.0, 0.0]
                    )

    def test_mb_acmm_endmembers(self):
        aug = named_endmember(mb50NCKFMASHTO.aug(), "acmm")
        dio = named_endmember(mb50NCKFMASHTO.dio(), "acmm")
        for mineral, shift in ((aug, -5000.0), (dio, -7000.0)):
            assert_linear_correction(
                self, mineral, [HP_2011_ds62.acm()], [1.0], [shift, 0.0, 0.0]
            )
        self.assertFloatEqual(aug.molar_gibbs - dio.molar_gibbs, 2000.0, tol=1.0e-12)

    def test_ig_cfm_endmembers(self):
        for dataset in (ig50NCKFMASHTOCr, ig50NCKFMASTOCr):
            with self.subTest(dataset=dataset.__name__):
                ol = named_endmember(dataset.ol(), "cfm")
                cpx = named_endmember(dataset.cpx(), "cfm")
                self.assertEqual(ol.formula, {"Mg": 1, "Fe": 1, "Si": 1, "O": 4})
                self.assertEqual(cpx.formula, {"Mg": 1, "Fe": 1, "Si": 2, "O": 6})
                assert_linear_correction(
                    self,
                    ol,
                    [HGP_2018_ds633.fa(), HGP_2018_ds633.fo()],
                    [0.5, 0.5],
                    [0.0, 0.0, 0.0],
                )
                assert_linear_correction(
                    self,
                    cpx,
                    [HGP_2018_ds633.en(), HGP_2018_ds633.fs()],
                    [0.5, 0.5],
                    [-1600.0, 2.0, 4.65e-7],
                )

    def test_ig_fluid_activities_and_water_endmember(self):
        fluid = ig50NCKFMASHTOCr.fl()
        water_index = fluid.endmember_names.index("H2O")
        self.assertEqual(fluid.n_endmembers, 11)
        self.assertEqual(
            native_endmembers(fluid)[water_index][0].formula,
            HP_2011_fluids.H2O().formula,
        )
        proportions = np.full(fluid.n_endmembers, 0.05)
        proportions[water_index] = 0.5
        expected = (1.0 - proportions[water_index]) * proportions
        expected[water_index] = proportions[water_index] ** 2
        self.assertArraysAlmostEqual(
            ideal_model_call(
                fluid.solution_model,
                "compute_activities",
                1000000000.0,
                1000.0,
                proportions,
            ),
            expected,
        )
        for pure in np.eye(fluid.n_endmembers):
            self.assertFloatEqual(
                fluid.solution_model.compute_excess_gibbs_free_energy(
                    1000000000.0, 1000.0, pure
                ),
                0.0,
                tol_zero=1e-09,
            )
        fluid.set_composition(proportions)
        fluid.set_state(1000000000.0, 1000.0)
        self.assertTrue(np.isfinite(fluid.molar_gibbs))
        self.assertGreater(fluid.molar_volume, 0.0)
        self.assertTrue(np.isfinite(fluid.molar_heat_capacity_p))
        pure_water = np.zeros(fluid.n_endmembers)
        pure_water[water_index] = 1.0
        fluid.set_composition(pure_water)
        water = HP_2011_fluids.H2O()
        water.set_state(1000000000.0, 1000.0)
        self.assertFloatEqual(fluid.molar_gibbs, water.molar_gibbs)
        self.assertFloatEqual(fluid.molar_volume, water.molar_volume)

    def test_melt_ideal_activities(self):
        for dataset, constructor in (
            (mb50NCKFMASHTO, mb50NCKFMASHTO.L),
            (mp50KFMASH, mp50KFMASH.liq),
            (mp50MnNCKFMASHTO, mp50MnNCKFMASHTO.liq),
            (mp50NCKFMASHTO, mp50NCKFMASHTO.liq),
        ):
            with self.subTest(dataset=dataset.__name__):
                melt = constructor()
                names = melt.endmember_names
                water = names.index("h2oL")
                magnesium = names.index("fo2L")
                iron = names.index("fa2L")
                proportions = np.arange(1.0, melt.n_endmembers + 1.0)
                proportions /= sum(proportions)
                dry_fraction = 1.0 - proportions[water]
                olivine_fraction = proportions[magnesium] + proportions[iron]
                expected = dry_fraction * proportions
                expected[water] = proportions[water] ** 2
                for i in (magnesium, iron):
                    expected[i] = (
                        dry_fraction
                        * olivine_fraction
                        * (proportions[i] / olivine_fraction) ** 5
                    )
                model = melt.solution_model
                self.assertArraysAlmostEqual(
                    ideal_model_call(
                        model, "compute_activities", 1000000000.0, 1500.0, proportions
                    ),
                    expected,
                )
                self.assertArraysAlmostEqual(
                    ideal_model_call(
                        model,
                        "compute_excess_partial_entropies",
                        1000000000.0,
                        1500.0,
                        proportions,
                    ),
                    -gas_constant * np.log(expected),
                )
                for pure in np.eye(melt.n_endmembers):
                    self.assertFloatEqual(
                        model.compute_excess_gibbs_free_energy(
                            1000000000.0, 1500.0, pure
                        ),
                        0.0,
                        tol_zero=1e-09,
                    )
                melt.set_composition(proportions)
                melt.set_state(1000000000.0, 1500.0)
                self.assertTrue(np.isfinite(melt.molar_gibbs))
                self.assertTrue(melt.molar_volume > 0.0)

    def test_ig50NCKFMASHTOCr(self):
        for m in dir(ig50NCKFMASHTOCr):
            mineral = getattr(ig50NCKFMASHTOCr, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)

    def test_ig50NCKFMASTOCr(self):
        for m in dir(ig50NCKFMASTOCr):
            mineral = getattr(ig50NCKFMASTOCr, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)

    def test_mb50NCKFMASHTO(self):
        for m in dir(mb50NCKFMASHTO):
            mineral = getattr(mb50NCKFMASHTO, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)

    def test_mp50KFMASH(self):
        for m in dir(mp50KFMASH):
            mineral = getattr(mp50KFMASH, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)

    def test_mp50MnNCKFMASHTO(self):
        for m in dir(mp50MnNCKFMASHTO):
            mineral = getattr(mp50MnNCKFMASHTO, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)

    def test_mp50NCKFMASHTO(self):
        for m in dir(mp50NCKFMASHTO):
            mineral = getattr(mp50NCKFMASHTO, m)
            if (
                is_native_factory(mineral)
                and mineral != Mineral
                and (mineral != CombinedMineral)
                and is_native_factory(mineral, burnman.Material)
            ):
                if is_native_factory(mineral, Solution) and mineral is not Solution:
                    mbr_names = mineral().endmember_names
                    n_mbrs = len(mbr_names)
                    molar_fractions = np.ones(n_mbrs) / n_mbrs
                    m = mineral()
                    m.set_composition(molar_fractions)
                    self.assertTrue(type(m.formula) is dict)
