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

"""Native mineral catalogues, named by publication year.

HP11 uses dataset 6.2; HGP18 and IG18 use dataset 6.33. HPx_ds636,
IG24 and IG25 provide dataset 6.36 and the W24/G25 igneous families.
MP14 and MB16 provide metapelite and metabasite solutions, including melts.
SLB11, JH15 and SLB24 provide mantle models and relaxed spin solutions.
Factories create independent native objects without Python BurnMan.
SLB24.phase_aliases maps original model names to preferred plot abbreviations.
model_sets provides matched collections with calibration metadata.
load_dataset selects candidate models for equilibrium and phase discovery.
"""

from importlib import import_module as _import_module
import sys as _sys

from .. import _core
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
    DKS_2013_solids,
    DKS_2013_liquids,
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
    "DKS_2013_solids",
    "DKS_2013_liquids",
    "model_sets",
]

for _name in __all__:
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]


def load_dataset(dataset, phase_names=None):
    """Return candidate models and metadata describing their selection.

    Supply a mineral module (e.g. SLB24), its name ("SLB24"), or a calibrated
    ModelSet. For example, ``phases, provenance = load_dataset(SLB24)``.
    Automatic selection collapses aliases, selects ordinary solution models
    and omits pure endmembers already represented within those solutions.
    Equilibrium solvers determine the spin populations of the ordinary models.

    phase_names optionally selects named factories in an explicit order,
    overriding automatic selection. It cannot override a ModelSet's models.
    Returned metadata records the dataset and selected factory names, or the
    ModelSet's calibration metadata, so a saved calculation can be reproduced.
    """
    if isinstance(dataset, str):
        dataset = _import_module(f"burnman_cpp.minerals.{dataset}")
    if hasattr(dataset, "phases"):
        if phase_names is not None:
            raise ValueError("phase_names cannot override a ModelSet.")
        return list(dataset.phases), dataset.to_dict()
    if phase_names is not None:
        names = list(phase_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("phase_names must be nonempty and unique.")
        return [getattr(dataset, name)() for name in names], dict(
            dataset=dataset.__name__, phase_names=names
        )
    selected = {}
    for name in sorted(vars(dataset)):
        factory = getattr(dataset, name)
        if name.startswith("_") or not callable(factory):
            continue
        # Dataset exports include constructors as well as mineral factories.
        if getattr(factory, "__module__", None) != dataset.__name__:
            continue
        material = factory()
        if isinstance(material, _core.RelaxedSolution):
            continue
        selected.setdefault(material.name, (name, material))
    endmembers = {
        name
        for _, phase in selected.values()
        if isinstance(phase, _core.Solution)
        for name in phase.endmember_names
    }
    selected = [
        (name, phase)
        for name, phase in selected.values()
        if isinstance(phase, _core.Solution) or phase.name not in endmembers
    ]
    if not selected:
        raise ValueError("The dataset contains no candidate mineral factories.")
    return [p for _, p in selected], dict(
        dataset=dataset.__name__, phase_names=[n for n, _ in selected]
    )


__all__ += ["load_dataset"]
