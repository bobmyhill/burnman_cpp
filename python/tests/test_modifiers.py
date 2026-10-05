import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params


def test_linear_modifier_and_cache_reset():
    mineral = bm.Mineral(oxide_params())
    mineral.set_state(3.0e9, 1200.0)
    g, s, v = mineral.molar_gibbs, mineral.molar_entropy, mineral.molar_volume
    mineral.set_property_modifiers(
        [("linear", dict(delta_E=1200.0, delta_S=5.0, delta_V=1.0e-7))]
    )
    assert mineral.molar_gibbs == pytest.approx(
        g + 1200.0 - 1200.0 * 5.0 + 3.0e9 * 1.0e-7
    )
    assert mineral.molar_entropy == pytest.approx(s + 5.0)
    assert mineral.molar_volume == pytest.approx(v + 1.0e-7)
    assert mineral.get_property_modifiers()["dGdP"] == pytest.approx(1.0e-7)
    mineral.set_property_modifiers([])
    assert mineral.molar_gibbs == pytest.approx(g)
    with pytest.raises(ValueError, match="Unknown property modifier"):
        mineral.set_property_modifiers([("missing", {})])
    with pytest.raises(ValueError, match="finite"):
        mineral.set_property_modifiers(
            [("linear", dict(delta_E=np.nan, delta_S=0.0, delta_V=0.0))]
        )


def test_signed_combined_endmember_survives_solution_copy():
    mg, fe = bm.Mineral(oxide_params("Mg")), bm.Mineral(oxide_params("Fe"))
    mixed = bm.CombinedMineral([mg, fe], [0.5, 0.5], [2500.0, 2.0, 1.0e-7])
    # A signed combination reconstructs MgO, including the linear corrections.
    recovered = bm.CombinedMineral([mixed, fe], [2.0, -1.0])
    np.testing.assert_allclose(list(recovered.formula.values()), [1.0, 1.0])
    assert recovered.formula == {"Mg": 1.0, "O": 1.0}
    assert recovered.molar_mass == pytest.approx(mg.molar_mass)
    p, t = 3.0e9, 1200.0
    mg.set_state(p, t)
    recovered.set_state(p, t)
    expected_g = mg.molar_gibbs + 2.0 * (2500.0 - 2.0 * t + 1.0e-7 * p)
    assert recovered.molar_gibbs == pytest.approx(expected_g)
    model = bm.IdealSolution([(recovered, "[Mg]O"), (fe, "[Fe]O")])
    solution = bm.Solution(model, [1.0, 0.0])
    solution.set_state(p, t)
    assert solution.molar_gibbs == pytest.approx(expected_g)
    assert recovered.params.equation_of_state == bm.EOSType.Custom
