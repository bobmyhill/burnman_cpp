#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-1: Kilbourne Hole peridotite KLB-1.

Bulk oxide moles are transcribed from the figure footer (more digits than
Table S3-1). Atomic O specifies Fe3+/total Fe = 2 O / FeO; no redox buffer.
"""

from green2025 import main

OXIDES = dict(
    SiO2=38.494,
    Al2O3=1.776,
    CaO=2.824,
    MgO=50.566,
    FeO=5.886,
    K2O=0.01,
    Na2O=0.25,
    TiO2=0.1,
    O=0.096,
    Cr2O3=0.109,
)
REFERENCE_POINTS = [
    (1000, 35, "cpx g opx ol"),
    (1000, 18, "g cpx opx ol sp"),
    (1000, 12, "cpx ol sp opx"),
    (1000, 5, "sp fsp cpx opx ol"),
    (1950, 25, "melt"),
]

if __name__ == "__main__":
    main(
        "klb1",
        "KLB-1 — Kilbourne Hole peridotite",
        1,
        OXIDES,
        (800, 2000),
        (0, 50),
        REFERENCE_POINTS,
        (25, 1890),  # Liquidus read from the published raster, approximately ±5 C.
    )
