#!/usr/bin/env python3
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

"""Harzburgite–basalt P-X section, SLB2024 solids, 1673 K and 0–25 GPa.

Table 1 of Stixrude & Lithgow-Bertelloni (2024), doi:10.1093/gji/ggae126,
supplies both dry oxide compositions, including their different ferric ratios.
Each native Composition is normalized to 100 g (0.1 kg). X is the basalt mass
fraction: bulk(X)=(1-X)*harzburgite+X*basalt, always representing 100 g.
The pyrolite example supplies the complete SLB2024 candidate set and settings.
All stability searches, equilibrium and tracing run in C++; Python plots.
The pyrolite EOS-domain checks are retained. Melting is excluded; oxygen is
closed and unbuffered.

Run: python examples/example_harzburgite_basalt_px.py
"""

from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    load_example,
    plot_phase_diagram_example,
    run_phase_diagram_example,
)

pyrolite = load_example(Path(__file__).with_name("example_pyrolite_phase_diagram.py"))
PAPER_URL = pyrolite.PAPER_URL
candidate_phases = pyrolite.candidate_phases
settings = pyrolite.settings


def endmember(oxides):
    composition = bm.Composition(oxides, unit_type="mass")
    composition.renormalize("mass", "total", 0.1)
    return composition


HARZBURGITE = endmember(
    dict(
        SiO2=43.43,
        MgO=45.93,
        FeO=8.34,
        Fe2O3=0.09,
        CaO=0.90,
        Al2O3=1.00,
        Na2O=0.01,
        Cr2O3=0.30,
    )
)
BASALT = endmember(
    dict(
        SiO2=50.42,
        MgO=9.77,
        FeO=7.10,
        Fe2O3=1.07,
        CaO=12.54,
        Al2O3=16.80,
        Na2O=2.23,
        Cr2O3=0.07,
    )
)


def calculate(seeds=7, quick=False, verbose=False):
    opts = settings(seeds, quick, verbose)
    opts.composition_seeds = opts.pressure_seeds
    return bm.phase_diagram(
        HARZBURGITE.atomic_composition,
        candidate_phases(),
        diagram="PX",
        composition_end=BASALT.atomic_composition,
        pressure_range=(0.0, 25.0e9),
        temperature=1673.0,
        settings=opts,
    )


def save_json(result, path):
    bm.save_phase_diagram(
        result,
        path,
        metadata=dict(
            endpoint_masses_kg={
                "harzburgite": HARZBURGITE.mass_composition,
                "basalt": BASALT.mass_composition,
            },
            composition_source=PAPER_URL,
            closed_system=True,
        ),
    )


def plot(result, path, label_fontsize=8.0):
    plot_phase_diagram_example(
        result,
        path,
        label_fontsize,
        formats=("pdf",),
        pressure_unit="GPa",
        composition_label="Basalt mass fraction",
        axes=dict(title="Harzburgite–basalt at 1673 K (SLB2024 solids)"),
        key_path=path.with_name(path.stem + "_labels.md"),
    )


def main():
    parser = example_parser(
        __doc__,
        Path(__file__).resolve().parent / "harzburgite_basalt_px_output",
        seeds=7,
        label_fontsize=8.0,
        replot=True,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.seeds, args.quick, args.verbose),
        stem="harzburgite_basalt_px",
        save_json=save_json,
        plot=plot,
    )


if __name__ == "__main__":
    main()
