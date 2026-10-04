from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from burnman_cpp.minerals import HP_2011_ds62 as HP, SLB_2011 as SLB, JH_2015 as JH

FACTORIES = [
    HP.sill,
    HP.andalusite,
    HP.ky,
    SLB.mg_fe_olivine,
    SLB.mg_fe_wadsleyite,
    SLB.mg_fe_ringwoodite,
    SLB.mg_fe_bridgmanite,
    SLB.post_perovskite,
    SLB.ferropericlase,
    SLB.orthopyroxene,
    SLB.garnet,
    SLB.ca_perovskite,
    SLB.pyrope_grossular,
    SLB.mg_fe_bridgmanite_binary,
    JH.orthopyroxene,
    JH.mg_fe_orthopyroxene,
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
    opx = JH.mg_fe_orthopyroxene()
    garnet = SLB.pyrope_grossular()
    bdg = SLB.mg_fe_bridgmanite_binary()
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
