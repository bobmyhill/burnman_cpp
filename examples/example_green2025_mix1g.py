#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-4: synthetic pyroxenite MIX1G (dry).

The bulk is in oxide moles, with a fixed inventory of excess atomic O.
"""

from green2025 import main

OXIDES = dict(
    SiO2=45.25,
    Al2O3=8.89,
    CaO=12.22,
    MgO=24.68,
    FeO=6.45,
    K2O=0.03,
    Na2O=1.39,
    TiO2=0.67,
    O=0.11,
    Cr2O3=0.02,
)
REFERENCE_POINTS = [
    (700, 15, "cpx sp ilm ol g"),
    (950, 15, "cpx sp g ol"),
    (1250, 18, "g sp cpx"),
    (1350, 6, "ol melt"),
    (1550, 10, "melt"),
]

if __name__ == "__main__":
    main(
        "mix1g",
        "MIX1G — synthetic pyroxenite",
        4,
        OXIDES,
        (600, 1600),
        (0, 20),
        REFERENCE_POINTS,
        (10, 1454),  # Liquidus read from the published raster, approximately ±5 C.
    )
