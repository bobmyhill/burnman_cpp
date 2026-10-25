# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Additive anharmonic lattice contribution assembled from native component models."""

from .. import _core, mineral_params


class AnharmonicDebye:
    """F=A(V/V_0)*[F_a(T,Theta(V))-F_a(T_0,Theta(V))], in molar SI units.

    Parameters may be a dictionary or MineralParams. Model entries must use
    the native characteristic-temperature, prefactor and thermal classes.
    """

    @staticmethod
    def validate_parameters(params):
        _core.anharmonic_debye.AnharmonicDebye.validate_parameters(
            mineral_params(params)
        )

    @staticmethod
    def helmholtz_energy(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.helmholtz_energy(
            temperature, volume, mineral_params(params)
        )

    @staticmethod
    def pressure(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.pressure(
            temperature, volume, mineral_params(params)
        )

    @staticmethod
    def isothermal_bulk_modulus(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.isothermal_bulk_modulus(
            temperature, volume, mineral_params(params)
        )

    @staticmethod
    def entropy(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.entropy(
            temperature, volume, mineral_params(params)
        )

    @staticmethod
    def heat_capacity_v(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.heat_capacity_v(
            temperature, volume, mineral_params(params)
        )

    @staticmethod
    def dSdV(temperature, volume, params):
        return _core.anharmonic_debye.AnharmonicDebye.dSdV(
            temperature, volume, mineral_params(params)
        )


__all__ = ["AnharmonicDebye"]
