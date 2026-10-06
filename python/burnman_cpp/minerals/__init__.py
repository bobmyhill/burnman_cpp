"""Native mineral catalogues, named by publication year.

HP11 uses dataset 6.2; HGP18 and IG18 use dataset 6.33. HPx_ds636,
IG24 and IG25 provide dataset 6.36 and the W24/G25 igneous families.
MP14 and MB16 provide metapelite and metabasite solutions, including melts.
SLB11, JH15 and SLB24 provide mantle models and relaxed spin solutions.
Factories create independent native objects without Python BurnMan.
model_sets provides matched collections with calibration metadata.

"""

import sys as _sys

from .._core.minerals import (
    HP11,
    SLB11,
    JH15,
    MP14,
    MB16,
    HGP18,
    IG18,
    HPx_ds636,
    IG24,
    IG25,
    SLB24,
    model_sets,
)

__all__ = [
    "HP11",
    "SLB11",
    "JH15",
    "MP14",
    "MB16",
    "HGP18",
    "IG18",
    "HPx_ds636",
    "IG24",
    "IG25",
    "SLB24",
    "model_sets",
]

for _name in __all__:
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]
