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

"""Native mineral catalogues with Python BurnMan's dataset names.

HP_2011_ds62 uses dataset 6.2; HGP_2018_ds633 and ig50NCKFMASHTOCr use
dataset 6.33. HPx_ds636, ig51W24 and ig51G25 provide dataset 6.36 and the
W24/G25 igneous families. mp50NCKFMASHTO and mb50NCKFMASHTO provide
metapelite and metabasite solutions, including melts.
mp50MnNCKFMASHTO provides Mn-bearing metapelite solutions.
SLB_2011, JH_2015 and SLB_2024 provide mantle models and relaxed spin solutions.
SLB_2005, SLB_2011_ZSB_2013 and SLB_2022 retain the original dataset names.
ICL_2018, KMFBZ_2017, Matas_etal_2007, Murakami_2013 and Murakami_etal_2012
provide core and mantle models. RS_2014_liquids supplies magnetic fayalite
liquid; Sundman_1991 provides magnetic iron phases. The other catalogue
contains additional solid and liquid iron models and ZSB2013 perovskites.
HHPH_2013 provides high-pressure endmembers; HP_2011_fluids provides water
and six gases. ig50NCKFMASTOCr provides dry igneous solutions, and mp50KFMASH
provides metapelite solutions. These catalogues retain their original names.
The older datasets have no reference free energies; SLB_2011_ZSB_2013 also
omits chemical formulae. Its elastic and thermal properties remain available.
Materials created from the same dataset are independent of one another.
HP_2011_ds62.cov(), HGP_2018_ds633.cov() and HPx_ds636.cov() provide
endmember energy covariances.
SLB_2024.phase_aliases maps original model names to preferred plot abbreviations.
model_sets provides matched collections with calibration metadata.
load_dataset selects candidate models for equilibrium and phase discovery.
"""

from importlib import import_module as _import_module
import sys as _sys

from .. import _core
from . import SE_2015, Pitzer_Sterner_1994
from .._core.minerals import (
    HP_2011_ds62,
    SLB_2011,
    JH_2015,
    mp50NCKFMASHTO,
    mb50NCKFMASHTO,
    HGP_2018_ds633,
    ig50NCKFMASHTOCr,
    HPx_ds636,
    ig51W24,
    ig51G25,
    SLB_2024,
    DKS_2013_solids,
    DKS_2013_liquids,
    SLB_2005,
    SLB_2011_ZSB_2013,
    SLB_2022,
    HHPH_2013,
    HP_2011_fluids,
    ig50NCKFMASTOCr,
    mp50KFMASH,
    mp50MnNCKFMASHTO,
    ICL_2018,
    KMFBZ_2017,
    Matas_etal_2007,
    Murakami_2013,
    Murakami_etal_2012,
    RS_2014_liquids,
    Sundman_1991,
    other,
    model_sets,
)

__all__ = [
    "other",
    "SE_2015",
    "Pitzer_Sterner_1994",
    "HP_2011_ds62",
    "SLB_2011",
    "JH_2015",
    "mp50NCKFMASHTO",
    "mb50NCKFMASHTO",
    "HGP_2018_ds633",
    "ig50NCKFMASHTOCr",
    "HPx_ds636",
    "ig51W24",
    "ig51G25",
    "SLB_2024",
    "DKS_2013_solids",
    "DKS_2013_liquids",
    "SLB_2005",
    "SLB_2011_ZSB_2013",
    "SLB_2022",
    "HHPH_2013",
    "HP_2011_fluids",
    "ig50NCKFMASTOCr",
    "mp50KFMASH",
    "mp50MnNCKFMASHTO",
    "ICL_2018",
    "KMFBZ_2017",
    "Matas_etal_2007",
    "Murakami_2013",
    "Murakami_etal_2012",
    "RS_2014_liquids",
    "Sundman_1991",
    "model_sets",
]

for _name in __all__:
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]


def _named_materials(dataset):
    for name in sorted(vars(dataset)):
        material_class = getattr(dataset, name)
        if name.startswith("_") or name == "cov" or not callable(material_class):
            continue
        if getattr(material_class, "__module__", None) != dataset.__name__:
            continue
        material = material_class()
        if isinstance(material, _core.Material) and not isinstance(
            material, _core.RelaxedSolution
        ):
            yield name, material


def available_phase_names(dataset):
    """Return all dataset names accepted by ``phase_names``, in sorted order.

    Supply a thermodynamic dataset, such as SLB_2024, or its name, "SLB_2024".
    The list includes pure endmembers and alternate names for materials.
    Relaxed Solution class objects are excluded because equilibrium determines
    their internal compositions. A ModelSet has a prescribed collection of
    materials and does not accept a separate ``phase_names`` selection.
    """
    if isinstance(dataset, str):
        dataset = _import_module(f"burnman_cpp.minerals.{dataset}")
    if hasattr(dataset, "phases"):
        raise ValueError("phase_names cannot override a ModelSet.")
    if not hasattr(dataset, "__name__"):
        raise TypeError("Supply a thermodynamic dataset or its name.")
    return [name for name, _ in _named_materials(dataset)]


def load_dataset(dataset, phase_names=None):
    """Return candidate models and metadata describing their selection.

    Supply a thermodynamic dataset (e.g. SLB_2024), its name ("SLB_2024"), or a calibrated
    ModelSet. For example, ``phases, provenance = load_dataset(SLB_2024)``.
    Automatic selection collapses aliases, selects ordinary solution models
    and omits pure endmembers already represented within those solutions.
    Equilibrium solvers determine the spin populations of the ordinary models.

    phase_names optionally selects named materials in an explicit order,
    overriding automatic selection. It cannot override a ModelSet's models.
    Returned metadata records the dataset and selected material names, or the
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
    for name, material in _named_materials(dataset):
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
        raise ValueError("The dataset contains no candidate Material class objects.")
    return [p for _, p in selected], dict(
        dataset=dataset.__name__, phase_names=[n for n, _ in selected]
    )


__all__ += ["available_phase_names", "load_dataset"]
