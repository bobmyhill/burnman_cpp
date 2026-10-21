#!/usr/bin/env python3
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
"""Shared SLB2024 Table 1 compositions and candidate catalogue for benchmarks."""

import gzip
import json

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24

TABLE1 = {
    "pyrolite": dict(
        SiO2=45.0,
        MgO=38.88,
        FeO=8.05,
        Fe2O3=0.18,
        CaO=3.18,
        Al2O3=4.0,
        Na2O=0.13,
        Cr2O3=0.57,
    ),
    "harzburgite": dict(
        SiO2=43.43,
        MgO=45.93,
        FeO=8.34,
        Fe2O3=0.09,
        CaO=0.90,
        Al2O3=1.0,
        Na2O=0.01,
        Cr2O3=0.30,
    ),
    "basalt": dict(
        SiO2=50.42,
        MgO=9.77,
        FeO=7.10,
        Fe2O3=1.07,
        CaO=12.54,
        Al2O3=16.80,
        Na2O=2.23,
        Cr2O3=0.07,
    ),
}
FERRIC_FRACTIONS = dict(pyrolite=0.02, harzburgite=0.01, basalt=0.12)


def bulk_composition(lithology):
    """100 g Table 1 bulk; preserve cation ratios and impose stated Fe3+/Fe.

    Table 1 oxide masses are rounded. Redistribute FeO/Fe2O3 to match the
    independently stated initial ferric fraction, then normalize to 100 g.
    Subsequent equilibrium calculations retain the closed elemental inventory.
    """
    masses = {k: v * 0.001 for k, v in TABLE1[lithology].items()}
    iron = bm.Composition({k: masses.pop(k) for k in ("FeO", "Fe2O3")}, "mass")
    total_fe, ferric = iron.atomic_composition["Fe"], FERRIC_FRACTIONS[lithology]
    bulk = bm.Composition(masses, "mass")
    bulk.add_components(
        dict(FeO=(1 - ferric) * total_fe, Fe2O3=0.5 * ferric * total_fe), "molar"
    )
    bulk.renormalize("mass", "total", 0.1)
    return bulk


def candidate_phases():
    """All ordinary SLB2024 solid solutions and competing pure phases."""
    factories = [
        ("ol", SLB24.olivine),
        ("wa", SLB24.wadsleyite),
        ("ri", SLB24.ringwoodite),
        ("opx", SLB24.orthopyroxene),
        ("cpx", SLB24.clinopyroxene),
        ("hpcpx", SLB24.c2c_pyroxene),
        ("gt", SLB24.garnet),
        ("sp", SLB24.mg_fe_aluminous_spinel),
        ("plg", SLB24.plagioclase),
        ("ak", SLB24.ilmenite),
        ("fp", SLB24.ferropericlase),
        ("bg", SLB24.bridgmanite),
        ("ppv", SLB24.post_perovskite),
        ("cf", SLB24.calcium_ferrite_structured_phase),
        ("nal", SLB24.new_aluminous_phase),
        ("capv", SLB24.capv),
        ("neph", SLB24.neph),
        ("ky", SLB24.ky),
        ("q", SLB24.qtz),
        ("coe", SLB24.coes),
        ("st", SLB24.st),
        ("seif", SLB24.apbo),
        ("wo", SLB24.wo),
        ("pwo", SLB24.pwo),
        ("Fe-bcc", SLB24.fea),
        ("Fe-fcc", SLB24.feg),
        ("Fe-hcp", SLB24.fee),
    ]
    phases = []
    for name, factory in factories:
        phase = factory()
        phase.set_name(name)
        phases.append(phase)
    return phases


def save_equilibrium_path(path, bulk, filename):
    """Retain equilibrium populations, active bases and diagnostic measures."""
    records = []
    for state in path.states:
        record = {
            k: getattr(state, k)
            for k in (
                "pressure",
                "temperature",
                "entropy",
                "volume",
                "success",
                "outside_model_domain",
                "message",
                "gibbs",
                "mass_balance_error",
                "minimum_affinity",
                "equilibrium_error",
            )
        }
        record["phases"] = [
            dict(
                id=p.id,
                candidate_index=p.candidate_index,
                name=p.name,
                amount=p.amount,
                composition=p.composition.tolist(),
                composition_basis=(
                    None
                    if p.composition_basis is None
                    else p.composition_basis.tolist()
                ),
            )
            for p in state.phases
        ]
        records.append(record)
    with gzip.open(filename, "wt", encoding="utf-8") as stream:
        json.dump(
            dict(bulk=bulk.atomic_composition, states=records), stream, allow_nan=False
        )
