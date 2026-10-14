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

"""A narrow P–T window in the dry closed-system SLB2024 pyrolite diagram.

Bulk composition, candidate models, EOS exclusions and plotting conventions
are shared with example_pyrolite_phase_diagram.py. Only the domain and seed
counts differ; all thermodynamic calculations use the same native routine.
"""

from functools import partial
from pathlib import Path

from burnman_cpp.tools.phase_diagram_helpers import (
    example_parser,
    load_example,
    run_phase_diagram_example,
)

pyrolite = load_example(Path(__file__).with_name("example_pyrolite_phase_diagram.py"))
PAPER_URL = pyrolite.PAPER_URL
INITIAL_FERRIC_FRACTION = pyrolite.INITIAL_FERRIC_FRACTION
PYROLITE_COMPOSITION = pyrolite.PYROLITE_COMPOSITION
PYROLITE_OXIDES = pyrolite.PYROLITE_OXIDES
PHASE_GLOSSARY = pyrolite.PHASE_GLOSSARY
pyrolite_composition = pyrolite.pyrolite_composition
candidate_phases = pyrolite.candidate_phases
refine = pyrolite.refine

P_Pa_low = 3.932e9
P_Pa_high = 3.935e9
T_K_low = 96.0
T_K_high = 98.0
DOMAIN = dict(
    pressure_range=(P_Pa_low, P_Pa_high), temperature_range=(T_K_low, T_K_high)
)
plot = partial(pyrolite.plot, **DOMAIN)
save_json = partial(pyrolite.save_json, **DOMAIN)

settings = pyrolite.settings


def calculate(seeds=9, quick=False, verbose=False):
    return pyrolite.calculate(seeds, quick, verbose, **DOMAIN)


def main():
    parser = example_parser(
        __doc__,
        "pyrolite_phase_diagram_zoom_output",
        seeds=9,
        replot=True,
        refine=True,
    )
    return run_phase_diagram_example(
        parser,
        lambda args: calculate(args.seeds, args.quick, args.verbose),
        stem="pyrolite_phase_diagram",
        save_json=save_json,
        plot=plot,
        refine=refine,
    )


if __name__ == "__main__":
    main()
