# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Compare the bundled one-dimensional seismic models in the lower mantle.

Run with ``python examples/example_seismic.py`` (requires matplotlib).
"""

import matplotlib.pyplot as plt
import numpy as np

from burnman_cpp import seismic
from burnman_cpp.tools.seismic import attenuation_correction


def main():
    depths = np.linspace(800.0e3, 2700.0e3, 200)
    names = ["PREM", "Slow", "Fast", "STW105", "IASP91", "AK135"]
    print("Model         Vp (m/s)    Vs (m/s)   Vphi (m/s) at 2000 km")
    for name in names:
        model = getattr(seismic, name)()
        vp, vs, vphi = model.evaluate(["v_p", "v_s", "v_phi"], [2000.0e3])[:, 0]
        print(f"{name:10s} {vp:11.3f} {vs:11.3f} {vphi:11.3f}")

    prem = seismic.prem_model
    pressure = prem.pressure(2000.0e3)
    print(f"\nPREM pressure at 2000 km: {pressure / 1.0e9:.3f} GPa")
    print(f"Depth from pressure: {prem.depth(pressure) / 1.0e3:.3f} km")
    corrected = attenuation_correction(12000.0, 7000.0, 9000.0, 200.0, 1000.0)
    print("Example attenuation-corrected velocities (m/s):", corrected)

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    for name in names:
        vp, vs = getattr(seismic, name)().evaluate(["v_p", "v_s"], depths)
        axes[0].plot(vp / 1.0e3, depths / 1.0e3, label=name)
        axes[1].plot(vs / 1.0e3, depths / 1.0e3, label=name)
    axes[0].set(xlabel="P-wave velocity (km/s)", ylabel="Depth (km)")
    axes[1].set(xlabel="S-wave velocity (km/s)")
    axes[0].invert_yaxis()
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
