# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Scalar spherical SSC through the native assemblage and composite APIs."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024
from burnman_cpp.tools.eos import check_eos_consistency
from conftest import assert_material_amount_scaling, oxide_params


def elastic_phase(bulk_gpa, shear_gpa):
    params = oxide_params()
    params.update(
        equation_of_state="bm3",
        K_0=bulk_gpa * 1.0e9,
        G_0=shear_gpa * 1.0e9,
    )
    return bm.Mineral(params)


@pytest.mark.parametrize(
    "porosity, expected",
    [
        (0.2, [37.15644096821864, 19.629142176318542]),
        (0.3, [22.689234722509436, 13.264359636280652]),
        (0.5, [0.007648199812123807, 0.005735868444317825]),
        (0.6, [5.976807619300475e-6, 4.848909539237201e-6]),
    ],
)
def test_scalar_python_reference_values(porosity, expected):
    # Python BurnMan's scalar SSC results for the Echoes spherical pore
    # parameters, in GPa. Equal endmember volumes give these volume fractions.
    rock = bm.Assemblage(
        [elastic_phase(72.0, 32.0), elastic_phase(1.0e-6, 1.0e-6)],
        [1.0 - porosity, porosity],
    )
    rock.set_averaging_scheme(bm.AveragingType.SelfConsistent)
    rock.set_state(0.0, 300.0)
    assert_allclose([rock.K_eff / 1.0e9, rock.G_eff / 1.0e9], expected, rtol=1.0e-10)
    assert rock.p_wave_velocity == pytest.approx(
        np.sqrt((rock.K_eff + 4.0 * rock.G_eff / 3.0) / rock.density)
    )
    assert rock.shear_wave_velocity == pytest.approx(np.sqrt(rock.G_eff / rock.density))


def test_single_phase_and_scheme_cache_invalidation():
    mineral = SLB_2024.periclase()
    pure = bm.Assemblage([mineral], [1.0])
    pure.set_averaging_scheme(bm.AveragingType.SelfConsistent)
    pure.set_state(120.0e9, 2500.0)
    assert pure.K_eff == pytest.approx(
        mineral.isentropic_bulk_modulus_reuss, rel=1.0e-12
    )
    assert pure.G_eff == pytest.approx(mineral.shear_modulus, rel=1.0e-12)
    assert pure.p_wave_velocity == pytest.approx(mineral.p_wave_velocity, rel=1.0e-12)

    rock = bm.Assemblage(
        [elastic_phase(4.0 / 3.0, 1.0), elastic_phase(16.0 / 3.0, 4.0)], [0.5, 0.5]
    )
    rock.set_state(0.0, 300.0)
    voigt_k = 0.5 * (4.0 / 3.0 + 16.0 / 3.0) * 1.0e9
    rock.set_averaging_scheme(bm.AveragingType.Voigt)
    assert rock.K_eff == pytest.approx(voigt_k)
    rock.set_averaging_scheme(bm.AveragingType.SelfConsistent)
    assert_allclose([rock.K_eff, rock.G_eff], [8.0e9 / 3.0, 2.0e9], rtol=1.0e-11)
    rock.set_fractions([1.0, 0.0])
    assert_allclose([rock.K_eff, rock.G_eff], [4.0e9 / 3.0, 1.0e9], rtol=1.0e-11)
    rock.set_averaging_scheme(bm.AveragingType.Voigt)
    rock.set_fractions([0.5, 0.5])
    assert rock.K_eff == pytest.approx(voigt_k)


def test_slb_solution_thermodynamics_amount_scaling_and_state_updates():
    bridgmanite = SLB_2024.bridgmanite()
    bridgmanite.set_composition([0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    phases = [bridgmanite, SLB_2024.periclase()]
    rock = bm.Assemblage(phases, [0.7, 0.3])
    for pressure, temperature in [(10.0e9, 1500.0), (120.0e9, 2500.0)]:
        rock.set_state(pressure, temperature)
        rock.set_averaging_scheme(bm.AveragingType.Voigt)
        upper = np.array([rock.K_eff, rock.G_eff])
        thermodynamics = np.array(
            [
                rock.isothermal_bulk_modulus_reuss,
                rock.isentropic_bulk_modulus_reuss,
                rock.density,
                rock.molar_entropy,
                rock.molar_gibbs,
            ]
        )
        rock.set_averaging_scheme(bm.AveragingType.Reuss)
        lower = np.array([rock.K_eff, rock.G_eff])
        rock.set_averaging_scheme(bm.AveragingType.SelfConsistent)
        actual = np.array([rock.K_eff, rock.G_eff])
        assert np.all(actual >= lower)
        assert np.all(actual <= upper)
        assert_allclose(
            [
                rock.isothermal_bulk_modulus_reuss,
                rock.isentropic_bulk_modulus_reuss,
                rock.density,
                rock.molar_entropy,
                rock.molar_gibbs,
            ],
            thermodynamics,
            rtol=1.0e-12,
        )
        assert check_eos_consistency(
            rock, P=pressure, T=temperature, including_shear_properties=False
        )
    assert_material_amount_scaling(rock)
    expected = np.array(
        [rock.K_eff, rock.G_eff, rock.p_wave_velocity, rock.shear_wave_velocity]
    )
    for amount in [0.1, 2.5]:
        copied = rock.copy()
        copied.number_of_moles = amount
        assert_allclose(
            [
                copied.K_eff,
                copied.G_eff,
                copied.p_wave_velocity,
                copied.shear_wave_velocity,
            ],
            expected,
            rtol=1.0e-10,
        )


@pytest.mark.parametrize("scheme", ["SelfConsistent", bm.AveragingType.SelfConsistent])
def test_equilibrated_composite_accepts_scheme(scheme):
    mineral = bm.Mineral(oxide_params())
    rock = bm.EquilibratedComposite({"MgO": 1.0}, [mineral], averaging_scheme=scheme)
    rock.set_state(4.0e9, 1000.0)
    mineral.set_state(4.0e9, 1000.0)
    assert rock.averaging_scheme == bm.AveragingType.SelfConsistent
    assert rock.K_eff == pytest.approx(mineral.isentropic_bulk_modulus_reuss)
    assert rock.G_eff == pytest.approx(mineral.shear_modulus)
