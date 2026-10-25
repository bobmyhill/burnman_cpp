#!/usr/bin/env python3
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

"""Export endmember energy covariances as HDF5 files with gzip compression.

Stored commits record export provenance. Audits compare values with the current
reference pin; exports preserve matching files and their original provenance.
"""

import argparse
from importlib import import_module
from pathlib import Path
import re

import h5py
import numpy as np

from burnman_reference import REFERENCE, ROOT, verify_reference

DATASETS = {"HP11": "HP_2011_ds62", "HGP18": "HGP_2018_ds633", "HPx_ds636": "HPx_ds636"}
DATA_DIRECTORY = Path("src/data/input_covariance")


def reference_data(dataset):
    cov = import_module(f"burnman.minerals.{dataset}").cov()
    names = np.asarray(cov["endmember_names"], dtype="S")
    matrix = np.asarray(cov["covariance_matrix"], dtype="<f8")
    if matrix.shape != (len(names), len(names)) or len(set(names)) != len(names):
        raise ValueError("Covariance dimensions must match unique endmember names.")
    if not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
        raise ValueError("Covariance matrices must be finite and symmetric.")
    return names, matrix


def attributes(dataset):
    return {
        "source_module": f"burnman.minerals.{dataset}_cov",
        "burnman_commit": REFERENCE["commit"],
        "units": "(J/mol)^2",
        "description": "Covariances of endmember zero-point energies",
        "license": "GPL-3.0-or-later",
        "copyright": "Copyright (C) 2025-2026 by the BurnMan Team.",
    }


def check_file(path, dataset, names, matrix):
    """Check values against the current pin, preserving the original export commit."""
    try:
        with h5py.File(path, "r") as data:
            values = data["covariance_matrix"]
            return (
                set(data) == {"endmember_names", "covariance_matrix"}
                and np.array_equal(data["endmember_names"][:], names)
                and np.array_equal(values[:].view(np.uint64), matrix.view(np.uint64))
                and values.compression == "gzip"
                and values.compression_opts == 9
                and values.shuffle
                and re.fullmatch(
                    r"[0-9a-f]{40}", str(data.attrs.get("burnman_commit", ""))
                )
                is not None
                and all(
                    data.attrs.get(key) == value
                    for key, value in attributes(dataset).items()
                    if key != "burnman_commit"
                )
            )
    except (OSError, KeyError, ValueError):
        return False


def main():
    import burnman

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_reference(burnman)
    mismatches = []
    for dataset in DATASETS.values():
        names, matrix = reference_data(dataset)
        filename = DATA_DIRECTORY / f"{dataset}_cov.h5"
        path = args.output_root / filename
        if args.check:
            if not check_file(path, dataset, names, matrix):
                mismatches.append(str(filename))
        else:
            if check_file(path, dataset, names, matrix):
                print(f"{filename}: unchanged")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w", libver="earliest") as data:
                data.attrs.update(attributes(dataset))
                data.create_dataset("endmember_names", data=names, track_times=False)
                data.create_dataset(
                    "covariance_matrix",
                    data=matrix,
                    chunks=(64, 64),
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                    track_times=False,
                )
            print(
                f"{filename}: {path.stat().st_size} bytes (matrix alone: {matrix.nbytes} bytes)"
            )
    if mismatches:
        raise SystemExit("Covariance data differs: " + ", ".join(mismatches))
    print(
        f"{'Verified' if args.check else 'Exported'} {len(DATASETS)} covariance datasets."
    )


if __name__ == "__main__":
    main()
