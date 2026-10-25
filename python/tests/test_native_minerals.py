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

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from burnman_cpp.minerals import (
    HP_2011_ds62,
    SLB_2011,
    JH_2015,
    HGP_2018_ds633,
    mp50MnNCKFMASHTO,
)

FACTORIES = [
    HGP_2018_ds633.iron,
    HGP_2018_ds633.wu,
    HGP_2018_ds633.mt,
    HGP_2018_ds633.hem,
    HP_2011_ds62.sill,
    HP_2011_ds62.andalusite,
    HP_2011_ds62.ky,
    SLB_2011.mg_fe_olivine,
    SLB_2011.mg_fe_wadsleyite,
    SLB_2011.mg_fe_ringwoodite,
    SLB_2011.mg_fe_bridgmanite,
    SLB_2011.post_perovskite,
    SLB_2011.ferropericlase,
    SLB_2011.orthopyroxene,
    SLB_2011.garnet,
    SLB_2011.ca_perovskite,
    SLB_2011.pyrope_grossular,
    SLB_2011.mg_fe_bridgmanite_binary,
    JH_2015.orthopyroxene,
    JH_2015.mg_fe_orthopyroxene,
    mp50MnNCKFMASHTO.mu,
    mp50MnNCKFMASHTO.bi,
    mp50MnNCKFMASHTO.g,
    mp50MnNCKFMASHTO.ilmm,
    mp50MnNCKFMASHTO.st,
]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_native_factories_independent_and_finite(factory):
    a, b = factory(), factory()
    a.set_state(3.0e9, 1200.0)
    gibbs = a.molar_gibbs
    formula = a.formula
    b.set_state(10.0e9, 1800.0)
    a.reset_cache()
    assert a.molar_gibbs == pytest.approx(gibbs, rel=1.0e-12)
    assert a.formula == formula
    assert np.isfinite([a.molar_gibbs, a.molar_entropy, a.molar_volume]).all()
    assert a.molar_volume > 0.0


def test_reduced_models_span_the_required_elements():
    opx = JH_2015.mg_fe_orthopyroxene()
    garnet = SLB_2011.pyrope_grossular()
    bdg = SLB_2011.mg_fe_bridgmanite_binary()
    assert opx.n_endmembers == 3
    assert garnet.n_endmembers == bdg.n_endmembers == 2
    assert set(opx.elements) == {"Mg", "Fe", "Si", "O"}
    assert set(garnet.elements) == {"Mg", "Ca", "Al", "Si", "O"}
    assert set(bdg.elements) == {"Mg", "Fe", "Si", "O"}
    opx.set_composition([0.55, -0.1, 0.55])
    opx.set_state(1.0e5, 800.0)
    assert np.isfinite(opx.molar_gibbs)
    with pytest.raises(ValueError, match="occupancies"):
        opx.set_composition([1.1, -1.2, 1.1])


def test_entire_example_imports_no_reference_or_optional_backends():
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "example_equilibrate.py"
    )
    # SciPy is required for potential-grid processing. The optional reference
    # and optimisation backends must still stay out of the native examples.
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(example)!r}, '--quick', '--no-plots']\n"
        f"runpy.run_path({str(example)!r}, run_name='__main__')\n"
        "blocked = {'burnman', 'cvxpy', 'sympy', 'cdd'}\n"
        "loaded = blocked.intersection(n.split('.')[0] for n in sys.modules)\n"
        "assert not loaded, f'Unexpected Python computation packages: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ringwoodite breakdown, fraction 1: 5/5 converged" in result.stdout
