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

"""Endmember covariance ordering, units and lossless HDF5 storage."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from burnman_cpp import minerals

DATASETS = [
    ("HP11", "HP_2011_ds62", 256, 286200.0, 5200.0),
    ("HGP18", "HGP_2018_ds633", 289, 286300.0, 5200.0),
    ("HPx_ds636", "HPx_ds636", 291, 289700.0, 5600.0),
]


@pytest.mark.parametrize("native,reference,n,fo_variance,reaction_variance", DATASETS)
def test_covariance_ordering_and_reaction_variance(
    native, reference, n, fo_variance, reaction_variance
):
    data = getattr(minerals, native).cov()
    assert set(data) == {"endmember_names", "covariance_matrix"}
    names, matrix = data["endmember_names"], data["covariance_matrix"]
    assert len(names) == len(set(names)) == n
    assert matrix.shape == (n, n)
    assert matrix.dtype == np.float64
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix[names.index("fo"), names.index("fo")] == fo_variance
    i, j = names.index("ky"), names.index("sill")
    assert matrix[i, i] + matrix[j, j] - 2.0 * matrix[i, j] == reaction_variance


@pytest.mark.parametrize("native", [entry[0] for entry in DATASETS])
def test_covariance_calls_return_independent_data(native):
    dataset = getattr(minerals, native)
    first, second = dataset.cov(), dataset.cov()
    first["endmember_names"][0] = "changed"
    first["covariance_matrix"][0, 0] = -1.0
    assert second["endmember_names"][0] == "fo"
    assert second["covariance_matrix"][0, 0] > 0.0
    assert not np.shares_memory(first["covariance_matrix"], second["covariance_matrix"])
    np.testing.assert_array_equal(
        dataset.cov()["covariance_matrix"], second["covariance_matrix"]
    )


@pytest.mark.parametrize("native", [entry[0] for entry in DATASETS])
def test_phase_selection_does_not_load_covariance(native, monkeypatch):
    dataset = getattr(minerals, native)

    def unavailable():
        raise AssertionError("Covariance data is not a phase.")

    unavailable.__module__ = dataset.__name__
    monkeypatch.setattr(dataset, "cov", unavailable)
    names = minerals.available_phase_names(dataset)
    assert "fo" in names and "cov" not in names


@pytest.mark.parametrize(
    "native,reference", [(entry[0], entry[1]) for entry in DATASETS]
)
def test_complete_covariance_matches_python_burnman(native, reference):
    pytest.importorskip("burnman")
    expected = importlib.import_module(f"burnman.minerals.{reference}").cov()
    actual = getattr(minerals, native).cov()
    assert actual["endmember_names"] == expected["endmember_names"]
    np.testing.assert_array_equal(
        actual["covariance_matrix"].view(np.uint64),
        expected["covariance_matrix"].view(np.uint64),
    )


@pytest.mark.parametrize(
    "native,reference", [(entry[0], entry[1]) for entry in DATASETS]
)
def test_hdf5_files_use_gzip_and_retain_exact_values(native, reference):
    h5py = pytest.importorskip("h5py")
    path = (
        Path(__file__).resolve().parents[2]
        / "src/data/input_covariance"
        / f"{reference}_cov.h5"
    )
    expected = getattr(minerals, native).cov()
    with h5py.File(path, "r") as data:
        matrix = data["covariance_matrix"]
        assert matrix.compression == "gzip" and matrix.compression_opts == 9
        assert matrix.shuffle
        assert data.attrs["units"] == "(J/mol)^2"
        assert list(data["endmember_names"].asstr()[:]) == expected["endmember_names"]
        np.testing.assert_array_equal(
            matrix[:].view(np.uint64), expected["covariance_matrix"].view(np.uint64)
        )


def test_covariance_export_check_rejects_changed_data(monkeypatch, tmp_path):
    pytest.importorskip("burnman")
    h5py = pytest.importorskip("h5py")
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "tools"))
    exporter = importlib.import_module("export_covariance_data")
    args = ["export_covariance_data.py", "--output-root", str(tmp_path)]
    monkeypatch.setattr("sys.argv", args)
    exporter.main()
    monkeypatch.setattr("sys.argv", args + ["--check"])
    exporter.main()
    path = tmp_path / exporter.DATA_DIRECTORY / "HP_2011_ds62_cov.h5"
    with h5py.File(path, "r+") as data:
        data["covariance_matrix"][0, 0] += 100.0
    with pytest.raises(SystemExit, match="Covariance data differs.*HP_2011_ds62"):
        exporter.main()
    with h5py.File(path) as data:
        assert data["covariance_matrix"][0, 0] == 286300.0
