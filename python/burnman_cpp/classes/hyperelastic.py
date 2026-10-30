# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Native finite-strain mineral thermodynamics and Cauchy-stress tangents."""

from .. import _core
from .._core import OutputFrame
from .anisotropicmineral import AnisotropicMineral


class HyperelasticMineral(_core.HyperelasticMineral):
    """Wrap a hydrostatic AnisotropicMineral with an isochoric Hencky model.

    Set the state with molar cell vectors (rows, metres) and temperature (K).
    Current/crystal output frames affect tensor components, preserving the
    supplied spatial cell. Cauchy-stress tangents need not have major symmetry.
    Thermal derivatives use the reference's central differences, with a smaller
    step below 0.2 K, and require positive temperature.
    """

    def __init__(self, hydrostatic_mineral):
        if isinstance(hydrostatic_mineral, _core.HyperelasticMineral):
            super().__init__(other=hydrostatic_mineral)
        else:
            super().__init__(hydrostatic_mineral)

    def copy(self):
        return type(self)(self)


# Reuse aliases through the single native inheritance chain.
for _name, _value in vars(AnisotropicMineral).items():
    if isinstance(_value, property):
        setattr(HyperelasticMineral, _name, _value)
