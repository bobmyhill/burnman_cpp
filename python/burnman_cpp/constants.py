"""Constants exported from the C++ library, using Python BurnMan's names.

gas_constant: molar gas constant R, J/(mol K).
Avogadro: Avogadro constant, 1/mol.
Boltzmann: Boltzmann constant, J/K.
G: Newtonian gravitational constant, m³/(kg s²).
Dirac: reduced Planck constant, J s.
invcm: conversion of 1 cm⁻¹ to J/mol.
logish_eps: fixed native logarithm regularization threshold.

Physical constants use the native CODATA 2022 values. invcm retains BurnMan's
tabulated precision. logish_eps is a float; assigning it does not change the
compiled solver's tolerance.
"""

from ._core.constants import (
    gas_constant,
    Avogadro,
    Boltzmann,
    G,
    Dirac,
    invcm,
    logish_eps,
)

__all__ = ["gas_constant", "Avogadro", "Boltzmann", "G", "Dirac", "invcm", "logish_eps"]
