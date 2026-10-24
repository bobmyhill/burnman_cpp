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

"""Dependency downloads recover from network failures without trusting bad data."""

import hashlib
from http.client import IncompleteRead
import importlib.util
import io
import os
from pathlib import Path
import socket
import tarfile
from urllib.error import URLError

import pytest


@pytest.fixture
def dependencies():
    path = Path(__file__).parents[2] / "tools/build_wheel_dependencies.py"
    spec = importlib.util.spec_from_file_location("build_wheel_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_archive(dependencies, monkeypatch, tmp_path):
    content = b"Verified dependency source\n"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        entry = tarfile.TarInfo("gmp-6.3.0/COPYING")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    data = output.getvalue()
    url = "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.gz"
    monkeypatch.setattr(
        dependencies, "SOURCES", {"gmp": (url, hashlib.sha256(data).hexdigest())}
    )
    return data, content, tmp_path / "gmp-gmp-6.3.0.tar.gz"


def test_gnu_dns_failure_uses_verified_mirror(
    dependencies, source_archive, monkeypatch, tmp_path
):
    data, content, archive = source_archive
    requests = []

    def urlopen(url, timeout):
        assert timeout > 0
        requests.append(url)
        if len(requests) == 1:
            raise URLError(socket.gaierror("Name resolution failed"))
        return io.BytesIO(data)

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", urlopen)
    source = dependencies.source_tree("gmp", tmp_path, os.environ)
    assert requests == [
        "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.gz",
        "https://ftpmirror.gnu.org/gmp/gmp-6.3.0.tar.gz",
    ]
    assert archive.read_bytes() == data
    assert (source / "COPYING").read_bytes() == content
    assert not archive.with_name(archive.name + ".part").exists()


@pytest.mark.parametrize("failure", ["dns", "interrupted"])
def test_download_retries_without_reusing_partial_data(
    dependencies, source_archive, monkeypatch, failure
):
    data, _, archive = source_archive
    temporary = archive.with_name(archive.name + ".part")
    temporary.write_bytes(b"stale partial download")
    requests = []
    delays = []

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise IncompleteRead(b"", len(data) - self.tell())
            return super().read(5)

    def urlopen(url, timeout):
        requests.append(url)
        if len(requests) == 1:
            if failure == "dns":
                raise URLError(socket.gaierror("Name resolution failed"))
            return InterruptedResponse(data)
        assert not temporary.exists()
        return io.BytesIO(data)

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dependencies.time, "sleep", delays.append)
    url = "https://example.invalid/dependency.tar.gz"
    dependencies.download_archive(url, archive, hashlib.sha256(data).hexdigest())
    assert requests == [url, url]
    assert delays == [1]
    assert archive.read_bytes() == data
    assert not temporary.exists()


def test_failed_downloads_are_bounded_and_never_extracted(
    dependencies, source_archive, monkeypatch, tmp_path
):
    _, _, archive = source_archive
    requests = []
    delays = []

    def urlopen(url, timeout):
        requests.append(url)
        raise URLError(socket.gaierror("Name resolution failed"))

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dependencies.time, "sleep", delays.append)
    with pytest.raises(RuntimeError, match="after 3 attempts") as failure:
        dependencies.source_tree("gmp", tmp_path, os.environ)
    assert isinstance(failure.value.__cause__, URLError)
    assert len(requests) == 6
    assert delays == [1, 2]
    assert not archive.exists()
    assert not archive.with_name(archive.name + ".part").exists()
    assert not (tmp_path / "gmp").exists()


@pytest.mark.parametrize("cached", [False, True])
def test_bad_checksum_prevents_extraction(
    dependencies, source_archive, monkeypatch, tmp_path, cached
):
    _, _, archive = source_archive
    requests = []

    def urlopen(url, timeout):
        requests.append(url)
        return io.BytesIO(b"Incorrect archive contents")

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", urlopen)
    if cached:
        archive.write_bytes(b"Incorrect cached contents")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        dependencies.source_tree("gmp", tmp_path, os.environ)
    assert len(requests) == (0 if cached else 1)
    assert archive.exists() == cached
    assert not archive.with_name(archive.name + ".part").exists()
    assert not (tmp_path / "gmp").exists()


def test_verified_cache_needs_no_network(
    dependencies, source_archive, monkeypatch, tmp_path
):
    data, content, archive = source_archive
    archive.write_bytes(data)

    def urlopen(*args, **kwargs):
        pytest.fail("A verified archive should be reused without downloading.")

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", urlopen)
    source = dependencies.source_tree("gmp", tmp_path, os.environ)
    assert (source / "COPYING").read_bytes() == content
