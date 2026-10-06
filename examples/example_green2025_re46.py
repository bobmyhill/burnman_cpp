#!/usr/bin/env python3
"""Green et al. (2025), Figure S3-2: Icelandic basalt RE46 (dry).

Oxide moles are transcribed from the figure footer. Atomic O is the excess
oxygen above FeO, so Fe3+/total Fe = 2 O / FeO in this closed system.
"""

from green2025 import main

OXIDES = dict(
    SiO2=50.4025,
    Al2O3=9.1027,
    CaO=15.1148,
    MgO=16.1483,
    FeO=7.0158,
    K2O=0.0099,
    Na2O=1.4608,
    TiO2=0.3876,
    O=0.3478,
    Cr2O3=0.0099,
)
REFERENCE_POINTS = [
    (1050, 2, "fsp cpx opx ol ilm"),
    (1100, 6, "fsp cpx opx ol"),
    (1150, 11, "fsp cpx opx"),
    (1210, 6, "fsp cpx opx melt"),
    (1300, 10, "cpx melt"),
    (1370, 5, "melt"),
]

if __name__ == "__main__":
    main(
        "re46",
        "RE46 — Icelandic basalt",
        2,
        OXIDES,
        (1000, 1400),
        (0, 12),
        REFERENCE_POINTS,
        (6, 1279),  # Liquidus read from the published raster, approximately ±5 C.
    )
