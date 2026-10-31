"""Seismic averaging stays separate from thermodynamic aggregate derivatives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import burnman_cpp as bm


def test_effective_moduli_follow_averaging_and_invalidate_cached_values():
    rock = bm.Assemblage(
        [bm.minerals.SLB_2005.wuestite(), bm.minerals.SLB_2005.mg_perovskite()],
        [0.5, 0.5],
    )
    with pytest.raises(RuntimeError):
        _ = rock.K_eff
    rock.set_state(40e9, 2000.0)
    thermodynamic = rock.isentropic_bulk_modulus_reuss
    phase_bulk = np.array([p.isentropic_bulk_modulus_reuss for p in rock.phases])
    phase_shear = np.array([p.shear_modulus for p in rock.phases])
    for fractions in ([0.5, 0.5], [0.2, 0.8]):
        rock.set_fractions(fractions)
        phase_volumes = rock.phase_volumes
        volume_fractions = rock.volume_fractions
        assert_allclose(
            phase_volumes, np.array(fractions) * [p.molar_volume for p in rock.phases]
        )
        assert_allclose(phase_volumes.sum(), rock.molar_volume)
        assert_allclose(volume_fractions.sum(), 1.0, rtol=1e-15)
        assert_allclose(volume_fractions, phase_volumes / phase_volumes.sum())
        for scheme in (bm.AveragingType.Voigt, bm.AveragingType.Reuss):
            rock.set_averaging_scheme(scheme)
            expected = [
                (
                    volume_fractions @ values
                    if scheme == bm.AveragingType.Voigt
                    else 1.0 / (volume_fractions @ (1.0 / values))
                )
                for values in (phase_bulk, phase_shear)
            ]
            assert_allclose([rock.K_eff, rock.G_eff], expected, rtol=1e-14)
            assert_allclose(
                rock.evaluate(
                    ["effective_isentropic_bulk_modulus", "effective_shear_modulus"],
                    40e9,
                    2000.0,
                ),
                expected,
                rtol=1e-14,
            )
            assert_allclose(rock.evaluate(["K_eff", "G_eff"], 40e9, 2000.0), expected)
            assert_allclose(rock.K_eff, rock.density * rock.bulk_sound_velocity**2)
            assert_allclose(rock.G_eff, rock.density * rock.shear_wave_velocity**2)
            if fractions == [0.5, 0.5]:
                assert rock.isentropic_bulk_modulus_reuss == thermodynamic
        rock.set_state(45e9, 2100.0)
        assert_allclose(rock.K_eff, rock.density * rock.bulk_sound_velocity**2)
        rock.set_state(40e9, 2000.0)


def test_single_mineral_effective_moduli_equal_scalar_elastic_moduli():
    mineral = bm.minerals.SLB_2011.periclase()
    for pressure, temperature in ((1e5, 300.0), (20e9, 1800.0)):
        mineral.set_state(pressure, temperature)
        assert (
            mineral.effective_isentropic_bulk_modulus
            == mineral.isentropic_bulk_modulus_reuss
        )
        assert mineral.effective_shear_modulus == mineral.shear_modulus
        assert_allclose(
            mineral.evaluate(["K_eff", "G_eff"], pressure, temperature),
            [mineral.K_eff, mineral.G_eff],
        )
