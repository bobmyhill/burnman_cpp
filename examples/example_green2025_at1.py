#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-3: high-Al basalt AT-1 (dry).

Use oxide moles, including atomic excess O, as in Table S3-1. Solvus branches
are generated internally; two feldspars/spinels remain distinct phases.
"""

from green2025 import main

OXIDES = dict(
    SiO2=54.4,
    Al2O3=12.96,
    CaO=11.31,
    MgO=7.68,
    FeO=8.63,
    K2O=0.54,
    Na2O=3.93,
    TiO2=0.79,
    O=0.41,
    Cr2O3=0.01,
)
REFERENCE_POINTS = [
    (925, 19.5, "fsp fsp g cpx q ru"),
    (900, 14, "fsp g cpx fsp ru"),
    (1100, 11.5, "cpx sp g fsp melt"),
    (1200, 5, "fsp melt"),
    (1300, 8, "melt"),
    (950, 6, "cpx fsp fsp sp sp ol ilm"),
]

if __name__ == "__main__":
    main(
        "at1",
        "AT-1 — high-Al basalt",
        3,
        OXIDES,
        (850, 1350),
        (0, 20),
        REFERENCE_POINTS,
        (10, 1283),  # Liquidus read from the published raster, approximately ±5 C.
    )
