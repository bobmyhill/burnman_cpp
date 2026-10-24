# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Build reproducible native wheel dependencies without using Homebrew bottles."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

SOURCES = {
    "eigen": (
        "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
        "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    ),
    "gmp": (
        "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz",
        "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898",
    ),
    "gsl": (
        "https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz",
        "6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190",
    ),
    "cddlib": (
        "https://github.com/cddlib/cddlib/releases/download/0.94n/cddlib-0.94n.tar.gz",
        "b87ee07ba2c1d0ab92a3e4eccacdf568f981a095a392e3b9efd7e7e4a9e125b1",
    ),
    "nlopt": (
        "https://github.com/stevengj/nlopt/archive/refs/tags/v2.10.1.tar.gz",
        "30d13ce16da119db3e987784f7864e35a562ec62c186352fae55cd003e6c58ff",
    ),
    "zlib": (
        "https://zlib.net/fossils/zlib-1.3.2.tar.gz",
        "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
    ),
    "hdf5": (
        "https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz",
        "e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b",
    ),
}


def source_tree(name, work, env):
    """Download a pinned archive, verify it, and extract its source directory."""
    url, checksum = SOURCES[name]
    archive = work / (name + "-" + url.rsplit("/", 1)[1])
    if not archive.exists():
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
        raise RuntimeError(f"Checksum mismatch: {archive}")
    source = work / name
    source.mkdir(exist_ok=True)
    # The HDF5 release archive has an additional leading './' component.
    strip_components = 2 if name == "hdf5" else 1
    subprocess.run(
        [
            "tar",
            "xf",
            str(archive),
            f"--strip-components={strip_components}",
            "-C",
            str(source),
        ],
        check=True,
        env=env,
    )
    return source


def build_dependencies(prefix, work, jobs):
    """Install shared libraries, Eigen headers and their notices into prefix."""
    prefix, work = prefix.resolve(), work.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    macos = platform.system() == "Darwin"
    target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0") if macos else ""
    compiler = ("clang", "clang++") if macos else ("gcc", "g++")
    manifest = dict(
        sources=SOURCES, arch=platform.machine(), target=target, compiler=compiler
    )
    stamp = prefix / "build-manifest.json"
    if stamp.exists():
        if json.loads(stamp.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError("Dependency settings changed; use a fresh prefix.")
        print(f"Using verified dependency build at {prefix}", flush=True)
        return
    flags = "-O2 -fPIC" + (f" -mmacosx-version-min={target}" if macos else "")
    env = dict(
        os.environ,
        CC=compiler[0],
        CXX=compiler[1],
        CFLAGS=flags,
        CXXFLAGS=flags,
        CPPFLAGS=f"-I{prefix}/include",
        LDFLAGS=f"-L{prefix}/lib -Wl,-rpath,{prefix}/lib",
    )
    if macos:
        env["MACOSX_DEPLOYMENT_TARGET"] = target

    def run(*args, cwd):
        print("+", *map(str, args), flush=True)
        subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

    for name in SOURCES:
        source = source_tree(name, work, env)
        if name in {"eigen", "nlopt", "zlib", "hdf5"}:
            options = {
                "eigen": ["-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF"],
                "nlopt": [
                    f"-DNLOPT_{feature}=OFF"
                    for feature in [
                        "PYTHON",
                        "OCTAVE",
                        "MATLAB",
                        "GUILE",
                        "JAVA",
                        "SWIG",
                        "TESTS",
                    ]
                ],
                "zlib": ["-DZLIB_BUILD_TESTING=OFF", "-DZLIB_BUILD_STATIC=OFF"],
                "hdf5": [
                    "-DBUILD_TESTING=OFF",
                    "-DBUILD_STATIC_LIBS=OFF",
                    "-DHDF5_BUILD_CPP_LIB=OFF",
                    "-DHDF5_BUILD_FORTRAN=OFF",
                    "-DHDF5_BUILD_JAVA=OFF",
                    "-DHDF5_BUILD_TOOLS=OFF",
                    "-DHDF5_BUILD_EXAMPLES=OFF",
                    "-DHDF5_BUILD_HL_LIB=OFF",
                    "-DHDF5_ENABLE_PARALLEL=OFF",
                    "-DHDF5_ENABLE_THREADSAFE=ON",
                    "-DHDF5_ENABLE_Z_LIB_SUPPORT=ON",
                    "-DHDF5_ENABLE_SZIP_SUPPORT=OFF",
                    "-DHDF5_ENABLE_ROS3_VFD=OFF",
                    f"-DZLIB_ROOT={prefix}",
                ],
            }[name]
            run(
                "cmake",
                "-S",
                source,
                "-B",
                source / "build",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=ON",
                *options,
                cwd=source,
            )
            run("cmake", "--build", source / "build", "--parallel", jobs, cwd=source)
            run("cmake", "--install", source / "build", cwd=source)
        else:
            # Modern libtool avoids flat-namespace libraries on recent macOS.
            if name in {"gmp", "cddlib"}:
                run("autoreconf", "-fi", cwd=source)
            options = ["--disable-assembly", "--with-pic"] if name == "gmp" else []
            run(
                source / "configure",
                f"--prefix={prefix}",
                "--enable-shared",
                "--disable-static",
                *options,
                cwd=source,
            )
            run("make", f"-j{jobs}", cwd=source)
            if name == "gmp":
                run("make", f"-j{jobs}", "check", cwd=source)
            run("make", "install", cwd=source)
        notices = prefix / "share/licenses" / name
        notices.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name.startswith(
                ("COPYING", "LICENSE", "AUTHORS", "COPYRIGHT")
            ):
                shutil.copy2(path, notices / path.name)
        (notices / "SOURCE.txt").write_text(SOURCES[name][0] + "\n")
    stamp.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build_dependencies(
        args.prefix,
        args.work_dir or args.prefix.parent / "burnman-wheel-build",
        args.jobs,
    )
