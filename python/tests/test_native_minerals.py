from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from burnman_cpp.minerals import (
    HP11,
    SLB11,
    JH15,
    HGP18,
)

FACTORIES = [
    HGP18.iron,
    HGP18.wu,
    HGP18.mt,
    HGP18.hem,
    HP11.sill,
    HP11.andalusite,
    HP11.ky,
    SLB11.mg_fe_olivine,
    SLB11.mg_fe_wadsleyite,
    SLB11.mg_fe_ringwoodite,
    SLB11.mg_fe_bridgmanite,
    SLB11.post_perovskite,
    SLB11.ferropericlase,
    SLB11.orthopyroxene,
    SLB11.garnet,
    SLB11.ca_perovskite,
    SLB11.pyrope_grossular,
    SLB11.mg_fe_bridgmanite_binary,
    JH15.orthopyroxene,
    JH15.mg_fe_orthopyroxene,
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
    opx = JH15.mg_fe_orthopyroxene()
    garnet = SLB11.pyrope_grossular()
    bdg = SLB11.mg_fe_bridgmanite_binary()
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


def test_entire_example_imports_no_reference_or_python_optimisation():
    example = (
        Path(__file__).resolve().parents[2] / "examples" / "example_equilibrate.py"
    )
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(example)!r}, '--quick', '--no-plots']\n"
        f"runpy.run_path({str(example)!r}, run_name='__main__')\n"
        "blocked = {'burnman', 'scipy', 'cvxpy', 'sympy', 'cdd'}\n"
        "loaded = blocked.intersection(n.split('.')[0] for n in sys.modules)\n"
        "assert not loaded, f'Unexpected Python computation packages: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ringwoodite breakdown, fraction 1: 5/5 converged" in result.stdout
