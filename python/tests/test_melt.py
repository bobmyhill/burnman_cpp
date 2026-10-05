import numpy as np
import pytest
from burnman_cpp.minerals import HGP_2018_ds633 as HGP


def test_native_melt_gibbs_composition_derivatives():
    s = HGP.silicate_melt()
    p = np.arange(1.0, s.n_endmembers + 1)
    p /= p.sum()
    s.set_composition(p)
    s.set_state(1e9, 1100.0)
    mu = s.partial_gibbs.copy()
    hessian = s.gibbs_hessian.copy()
    step = 1e-5
    for i in range(1, len(p)):
        high, low = p.copy(), p.copy()
        high[i] += step
        high[0] -= step
        low[i] -= step
        low[0] += step
        s.set_composition(high)
        g_high = s.molar_gibbs
        mu_high = s.partial_gibbs.copy()
        s.set_composition(low)
        g_low = s.molar_gibbs
        mu_low = s.partial_gibbs.copy()
        assert (g_high - g_low) / (2 * step) == pytest.approx(mu[i] - mu[0], abs=0.02)
        np.testing.assert_allclose(
            (mu_high - mu_low) / (2 * step),
            hessian[:, i] - hessian[:, 0],
            rtol=1e-6,
            atol=0.03,
        )
