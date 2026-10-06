#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-5: water-undersaturated tonalite 101.

The figure footer gives oxide mole amounts. Water (2.8005 mol% H2O) and
oxygen are closed inventories; the G25 fluid can appear where stable.
"""

from green2025 import main

OXIDES = dict(
    SiO2=63.0113,
    Al2O3=11.438,
    CaO=6.7338,
    MgO=3.9805,
    FeO=5.0503,
    K2O=1.4947,
    Na2O=3.9333,
    TiO2=0.6293,
    O=0.9283,
    H2O=2.8005,
)
REFERENCE_POINTS = [
    (620, 9.5, "hb bi ep fsp fsp sp cpx q ru"),
    (880, 5, "fsp cpx opx ilm sp melt"),
    (950, 5, "sp fsp opx melt"),
    (1100, 5, "fsp melt"),
    (1270, 5, "melt"),
]

if __name__ == "__main__":
    main(
        "ton101u",
        "ton101u — water-undersaturated tonalite 101",
        5,
        OXIDES,
        (600, 1300),
        (1, 10),
        REFERENCE_POINTS,
        (5, 1225),  # Liquidus read from the published raster, approximately ±5 C.
    )
