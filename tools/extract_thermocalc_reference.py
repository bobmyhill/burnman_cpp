#!/usr/bin/env python3
"""Check or extract energy benchmarks from the pinned HPx-eos archive.

No BurnMan implementation is imported or executed. Native tests use the
extracted header offline. Only units are converted: kbar to Pa, Celsius to K,
and kJ/mol to J/mol. The source archive's version files and SHA256 are checked.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "tests/reference/thermocalc_metapelite.json").read_text())
OUTPUT = ROOT / "tests/include/thermocalc_reference.hpp"


def extract(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != POLICY["sha256"]:
        raise ValueError("THERMOCALC archive does not match the pinned SHA256.")
    phases, endmembers = [], []
    with zipfile.ZipFile(archive) as source:
        paths = sorted(
            name
            for name in source.namelist()
            if name.endswith(".txt") and not name.startswith("__MACOSX/")
        )
        for name in paths:
            if name.endswith("/versions.txt"):
                continue
            version_path = name.rsplit("/", 1)[0] + "/versions.txt"
            version = source.read(version_path).decode("ascii").splitlines()
            if (
                len(version) != 3
                or version[0] != POLICY["thermocalc"]
                or version[2] != POLICY["dataset"]
            ):
                raise ValueError(f"Unexpected THERMOCALC version in {version_path}.")
            text = source.read(name).decode("latin-1")
            point = re.search(r"\{([\d.]+),\s*([\d.]+)\}", text)
            if point is None:
                raise ValueError(f"No pressure/temperature in {name}.")
            pressure, temperature = map(float, point.groups())
            pressure *= 1e8
            temperature += 273.15
            filename = Path(name).name
            properties = False
            activities = False
            phase = None
            for line in text.splitlines():
                fields = line.split()
                if fields == ["G", "H", "S", "V", "rho"]:
                    properties = True
                    continue
                if fields and fields[0] == "sys":
                    properties = False
                if properties and fields and fields[0] in POLICY["pure_phases"]:
                    if len(fields) != 6:
                        raise ValueError(f"Malformed phase energies in {name}: {line}")
                    phases.append(
                        (
                            filename,
                            pressure,
                            temperature,
                            fields[0],
                            float(fields[1]) * 1000,
                            float(fields[2]) * 1000,
                        )
                    )
                if "ideal" in fields and "gamma" in fields:
                    activities = True
                    continue
                if not activities or version[1] not in POLICY["endmember_model_files"]:
                    continue
                if len(fields) == 8:
                    phase, endmember = fields[:2]
                elif len(fields) == 7:
                    endmember = fields[0]
                else:
                    continue
                if phase in POLICY["excluded_endmember_phases"]:
                    continue
                if phase is None:
                    raise ValueError(f"Endmember has no parent phase in {name}.")
                endmembers.append(
                    (
                        filename,
                        pressure,
                        temperature,
                        phase,
                        endmember,
                        float(fields[-2]) * 1000,
                    )
                )
    if not phases or not endmembers:
        raise ValueError("No THERMOCALC energy benchmarks were extracted.")
    return phases, endmembers


def cpp_row(values):
    return (
        "{"
        + ", ".join(
            json.dumps(value) if isinstance(value, str) else repr(value)
            for value in values
        )
        + "}"
    )


def render(phases, endmembers):
    text = f"""/* Energy values extracted from the HPx-eos THERMOCALC benchmarks.
 * Source: {POLICY['archive']}
 * SHA256: {POLICY['sha256']}
 * Regenerate with tools/extract_thermocalc_reference.py --archive PATH --write.
 * tc350beta1, tc-ds62.txt. Energies in J/mol, pressure in Pa, temperature in K.
 * No Python or C++ BurnMan calculation contributes expected values.
 */
#pragma once
#include <array>
namespace thermocalc_reference {{
struct PhaseState {{
  const char *source;
  double pressure, temperature;
  const char *phase;
  double gibbs, enthalpy;
}};
struct EndmemberState {{
  const char *source;
  double pressure, temperature;
  const char *phase, *endmember;
  double gibbs;
}};
inline constexpr double energy_tolerance = {POLICY['energy_tolerance_J_per_mol']};
inline constexpr std::array<PhaseState, {len(phases)}> phases = {{{{
"""
    text += ",\n".join(cpp_row(row) for row in phases) + "\n}};\n"
    text += f"inline constexpr std::array<EndmemberState, {len(endmembers)}> endmembers = {{{{\n"
    text += ",\n".join(cpp_row(row) for row in endmembers)
    text += "\n}};\n} // namespace thermocalc_reference\n"
    return subprocess.check_output(
        ["clang-format", f"--assume-filename={OUTPUT}"], input=text, text=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        phases, endmembers = extract(args.archive)
    except ValueError as error:
        parser.error(str(error))
    generated = render(phases, endmembers)
    if args.write:
        OUTPUT.write_text(generated)
    elif OUTPUT.read_text() != generated:
        parser.exit(1, "THERMOCALC reference differs; regenerate with --write.\n")
    print(
        f"{'Wrote' if args.write else 'Verified'} {len(phases)} phase states "
        f"and {len(endmembers)} endmember energies from THERMOCALC."
    )


if __name__ == "__main__":
    main()
