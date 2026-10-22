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

"""Compare SLB periclase with volume-rescaled anharmonic modular MGD.

Based on BurnMan's examples/example_modular_mgd_with_anharmonicity.py.
This model uses a BM3 reference, the SLB Debye-temperature model, and
c_anh=0.04. It modifies the quasiharmonic framework through volume rescaling,
rather than adding the separate anharmonic term used in example_modular_mgd.py.

Run: python examples/example_modular_mgd_with_anharmonicity.py
Edit the settings below; this script takes no command-line arguments.
All calculations use burnman_cpp; Python BurnMan is not required.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as bm
from burnman_cpp import debye_temperature_models
from burnman_cpp.minerals.SLB11 import periclase

# Share the evaluation and seven-panel plotting layout with the first example.
from example_modular_mgd import plot_comparison, print_properties

TEMPERATURES = np.linspace(10.0, 3000.0, 101)
PRESSURES = np.array([0.0, 10.0, 30.0, 60.0, 100.0]) * 1.0e9
C_ANH = 0.04
OUTPUT_DIR = Path(__file__).with_name("modular_mgd_with_anharmonicity_output")
SHOW_PLOTS = True


def make_minerals():
    m_SLB = periclase()
    params = m_SLB.params
    params.equation_of_state = bm.EOSType.ModularMGDWithAnharmonicity
    params.reference_eos = bm.make_eos("bm3")
    params.debye_temperature_model = debye_temperature_models.SLB()

    # This is the simplified volume-rescaling model used by the Python example.
    # Its c_anh is not directly interchangeable with the parameter in the
    # original Wu and Wentzcovitch (2009) model.
    params.c_anh = C_ANH
    m = bm.Mineral(params)
    # SLB_2011 periclase has no property modifiers to copy.
    return m_SLB, m


def main():
    m_SLB, m = make_minerals()
    print_properties(m)
    fig, axes = plot_comparison(m_SLB, m, TEMPERATURES, PRESSURES)
    axes[1].set_ylim(0.0, 80.0)
    axes[2].set_ylim(0.0, 80.0)
    axes[3].set_ylim(0.0, 10.0)
    axes[4].set_ylim(1.0, 2.0)
    axes[5].set_ylim(100.0, 300.0)
    fig.suptitle(f"Periclase: volume-rescaled anharmonicity, $c_{{anh}} = {C_ANH}$")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"periclase_isobars.{extension}", dpi=180)
    print(f"Saved figures to {OUTPUT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
