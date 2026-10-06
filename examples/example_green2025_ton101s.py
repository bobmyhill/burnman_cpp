#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-6: water-saturated tonalite 101.

This is a closed bulk with 38.8982 mol% H2O, as in the figure footer, rather
than an imposed water chemical potential. Use the full G25 fluid with PS94
H2O and dissolved silicates, not an independently added pure-water phase.
"""

from green2025 import main

OXIDES = dict(
    SiO2=39.6103,
    Al2O3=7.1902,
    CaO=4.233,
    MgO=2.5022,
    FeO=3.1748,
    K2O=0.9396,
    Na2O=2.4726,
    TiO2=0.3956,
    O=0.5835,
    H2O=38.8982,
)
REFERENCE_POINTS = [
    (790, 8, "hb sp melt fluid"),
    (1100, 8, "melt fluid"),
    (840, 4, "hb fsp sp melt fluid"),
    (1050, 1.4, "fsp melt fluid"),
    (1250, 5, "melt fluid"),
]

if __name__ == "__main__":
    main(
        "ton101s",
        "ton101s — water-saturated tonalite 101",
        6,
        OXIDES,
        (600, 1300),
        (1, 10),
        REFERENCE_POINTS,
        (5, 1001),  # Liquidus read from the published raster, approximately ±5 C.
    )
