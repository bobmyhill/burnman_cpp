#!/usr/bin/env python3
"""Check or regenerate PS1994 snapshots from the pinned Python BurnMan branch.

The PS1994 implementation is absent from the main-branch reference. Supply a
clean checkout of the separate commit in tests/reference/ps1994.json. Expected
values use Python BurnMan's EOS functions; no native implementation is imported.
"""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
CPP_HEADER = (ROOT / "contrib/utilities/cpp_header.txt").read_text() + "\n"
POLICY = json.loads((ROOT / "tests/reference/ps1994.json").read_text())
OUTPUT = ROOT / "tests/include/ps1994_reference.hpp"
STATES = [
    (1e5, 500.0),
    (2e6, 500.0),
    (3e6, 500.0),
    (5e9, 500.0),
    (1e5, 1700.0),
    (5e9, 1700.0),
    (1e3, 1200.0),
    (1e6, 600.0),
    (4e6, 550.0),
    (8e6, 550.0),
    (5e7, 550.0),
    (1e7, 625.0),
    (2e7, 625.0),
    (25e6, 650.0),
    (1e8, 700.0),
    (1e9, 1000.0),
    (5e9, 1000.0),
    (5e9, 501.0),
    (5e9, 1699.0),
]


def load_reference(source):
    source = source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if commit != POLICY["commit"] or changed:
        raise ValueError(f"Reference must be a clean checkout of {POLICY['commit']}.")
    sys.path.insert(0, str(source))
    import burnman

    if Path(burnman.__file__).resolve() != source / "burnman/__init__.py":
        raise ValueError("Imported BurnMan does not belong to the supplied checkout.")
    module_name, mineral_name = POLICY["mineral"].rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), mineral_name)
    return burnman, factory()


def stable_state(eos, params, pressure, temperature):
    # The upstream volume() brackets the entire isotherm once and can return a
    # metastable root. Enumerate its pressure() roots independently and minimize
    # its Gibbs energy over roots with positive isothermal bulk modulus.
    upper = max(1.0, 10.0 * POLICY["gas_constant_J_per_mol_K"] * temperature / pressure)
    volumes = np.geomspace(1e-7, upper, 4097)
    residuals = eos.pressure(temperature, volumes, params) - pressure
    crossings = np.flatnonzero(np.signbit(residuals[:-1]) != np.signbit(residuals[1:]))
    stable = []
    for index in crossings:
        volume = brentq(
            eos._delta_pressure,
            volumes[index],
            volumes[index + 1],
            args=(pressure, temperature, params),
            xtol=POLICY["volume_root_xtol_m3_per_mol"],
        )
        bulk_modulus = eos.isothermal_bulk_modulus(
            pressure, temperature, volume, params
        )
        if bulk_modulus > 0.0:
            stable.append(
                (
                    eos.gibbs_free_energy(pressure, temperature, volume, params),
                    volume,
                    bulk_modulus,
                )
            )
    if not stable:
        raise ValueError(
            f"No stable Python reference root at P={pressure}, T={temperature}."
        )
    gibbs, volume, bulk_modulus = min(stable)
    values = (
        volume,
        bulk_modulus,
        eos.thermal_expansivity(pressure, temperature, volume, params),
        gibbs,
        eos.entropy(pressure, temperature, volume, params),
        eos.molar_heat_capacity_p(pressure, temperature, volume, params),
    )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Nonfinite Python reference at P={pressure}, T={temperature}."
        )
    return values, len(stable)


def generate(source):
    burnman, mineral = load_reference(source)
    # PS1994 uses this R in C++. Change only the in-memory parameter, restoring
    # it afterwards; the upstream EOS, coefficient table and checkout stay intact.
    previous_r = burnman.constants.gas_constant
    burnman.constants.gas_constant = POLICY["gas_constant_J_per_mol_K"]
    try:
        rows = []
        references = {}
        for pressure, temperature in STATES:
            if temperature not in references:
                references[temperature] = stable_state(
                    mineral.method,
                    mineral.params,
                    POLICY["reference_pressure_Pa"],
                    temperature,
                )[0]
            reference = references[temperature]
            values, roots = stable_state(
                mineral.method, mineral.params, pressure, temperature
            )
            numbers = [pressure, temperature, *values[:3]]
            numbers += [values[i] - reference[i] for i in range(3, 6)]
            rows.append(
                "{" + ", ".join(repr(float(x)) for x in numbers) + f", {roots}" + "}"
            )
    finally:
        burnman.constants.gas_constant = previous_r
    text = CPP_HEADER + f"""/* Generated from Python BurnMan; no C++ output is used.
 * Reference: {POLICY['repository']} at {POLICY['commit']}.
 * Regenerate with tools/generate_ps1994_reference.py --reference PATH --write.
 * R is matched to PS1994's {POLICY['gas_constant_J_per_mol_K']} J/(mol K).
 * Delta G, S and Cp are relative to {POLICY['reference_pressure_Pa']} Pa at the same T;
 * the upstream Debye thermal reference and native ideal-gas reference cancel.
 */
#pragma once
#include <array>

namespace ps1994_reference {{
struct State {{
  double pressure, temperature, volume, bulk_modulus, expansivity;
  double delta_gibbs, delta_entropy, delta_cp;
  int stable_roots;
}};
inline constexpr double reference_pressure = {POLICY['reference_pressure_Pa']};
inline constexpr std::array<State, {len(rows)}> states = {{{{
  """ + ",\n  ".join(rows) + "\n}};\n} // namespace ps1994_reference\n"
    # Use the repository's C++ formatting when producing and checking snapshots.
    return subprocess.check_output(
        ["clang-format", f"--assume-filename={OUTPUT}"], input=text, text=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Pinned Python BurnMan checkout"
    )
    parser.add_argument(
        "--write", action="store_true", help="Regenerate the native snapshot header"
    )
    args = parser.parse_args()
    generated = generate(args.reference)
    if args.write:
        OUTPUT.write_text(generated)
        print(f"Wrote {len(STATES)} Python BurnMan PS1994 reference states.")
    elif OUTPUT.read_text() != generated:
        parser.exit(
            1, "PS1994 snapshots differ from Python BurnMan; regenerate with --write.\n"
        )
    else:
        print(f"Verified {len(STATES)} Python BurnMan PS1994 reference states.")


if __name__ == "__main__":
    main()
