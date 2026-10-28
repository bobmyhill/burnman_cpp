# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Convert a gold pressure scale, including correlated P/T uncertainty."""

import numpy as np
from burnman_cpp import calibrants
from burnman_cpp.calibrants.tools import pressure_to_pressure


def main():
    old = calibrants.Anderson_1989.Au()
    new = calibrants.Fei_2007.Au()
    pressure, temperature = 20.0e9, 1000.0
    volume = old.volume(pressure, temperature)
    print(f"Measured volume: {volume:.9e} m3/mol")
    print(f"Original pressure: {old.pressure(volume, temperature)/1.e9:.6f} GPa")
    PT_covariance = np.array([[1.0e18, 1.0e8], [1.0e8, 100.0]])
    converted, covariance = pressure_to_pressure(
        old, new, pressure, temperature, PT_covariance
    )
    print(
        f"Revised pressure: {converted/1.e9:.6f} "
        f"+/- {np.sqrt(covariance[0, 0])/1.e9:.6f} GPa (1 sigma)"
    )
    print("Revised covariance in [pressure, temperature] order:")
    print(covariance)
    recovered = pressure_to_pressure(new, old, converted, temperature)
    np.testing.assert_allclose(recovered, pressure, rtol=1.0e-9)
    return converted, covariance


if __name__ == "__main__":
    main()
