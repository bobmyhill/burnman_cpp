#!/usr/bin/env python3
"""Check or regenerate PS1994 snapshots from the shared pinned Python BurnMan.

Expected values use Python BurnMan's EOS functions; no native implementation
is imported. The HP2011 thermal reference permits absolute-energy comparisons.
"""

import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq

from burnman_reference import REFERENCE, ROOT, verify_reference

CPP_HEADER = (ROOT / "contrib/utilities/cpp_header.txt").read_text() + "\n"
POLICY = REFERENCE | json.loads((ROOT / "tests/reference/ps1994.json").read_text())
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
# Match the numerical budgets of tests/minerals/test_water.cpp. Derivatives
# use finite differences upstream; their last digits vary across platforms.
COLUMNS = [
    ("pressure", 0.0, 0.0),
    ("temperature", 0.0, 0.0),
    ("volume", 2e-12, 0.0),
    ("bulk modulus", 2e-8, 0.0),
    ("expansivity", 2e-6, 0.0),
    ("delta Gibbs energy", 2e-11, 2e-8),
    ("delta entropy", 2e-7, 2e-7),
    ("delta heat capacity", 2e-5, 2e-3),
    ("Gibbs energy", 0.0, 2e-8),
    ("entropy", 2e-7, 2e-7),
    ("heat capacity", 2e-5, 2e-3),
    ("stable roots", 0.0, 0.0),
]


def format_header(text):
    return subprocess.check_output(
        ["clang-format", f"--assume-filename={OUTPUT}"], input=text, text=True
    )


def check_snapshot(stored, generated):
    """Check provenance and numeric states independently of formatter versions."""

    def parse(text):
        text = format_header(text)
        table = re.search(r"states\s*=\s*\{(.*?)\};", text, re.DOTALL)
        if table is None:
            raise ValueError("PS1994 snapshot state table is missing.")
        rows = re.findall(r"\{([^{}]+)\}", table[1])
        values = np.array([[float(x) for x in row.split(",")] for row in rows])
        if values.shape != (len(STATES), len(COLUMNS)):
            raise ValueError("PS1994 snapshot state table has the wrong shape.")
        metadata = text[: table.start(1)] + text[table.end(1) :]
        return re.sub(r"\s+", "", metadata), values

    stored_metadata, stored_values = parse(stored)
    generated_metadata, generated_values = parse(generated)
    if stored_metadata != generated_metadata:
        raise ValueError("PS1994 snapshot provenance or declarations differ.")
    for i, (name, relative, absolute) in enumerate(COLUMNS):
        np.testing.assert_allclose(
            stored_values[:, i],
            generated_values[:, i],
            rtol=relative,
            atol=absolute,
            err_msg=f"PS1994 {name}",
        )


def load_reference(source):
    source = source.resolve()
    sys.path.insert(0, str(source))
    import burnman

    if verify_reference(burnman) != source:
        raise ValueError("Imported BurnMan does not belong to the supplied checkout.")
    module_name, mineral_name = POLICY["mineral"].rsplit(".", 1)
    factory = getattr(importlib.import_module(module_name), mineral_name)
    return burnman, factory()


def stable_state(eos, params, pressure, temperature):
    # Independently enumerate roots to check upstream's stable-root selection
    # and record which states admit both liquid and vapour roots.
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
        bulk_modulus = eos.isothermal_bulk_modulus_reuss(
            pressure, temperature, volume, params
        )
        if bulk_modulus > 0.0:
            stable.append(
                (
                    eos.gibbs_energy(pressure, temperature, volume, params),
                    volume,
                    bulk_modulus,
                )
            )
    if not stable:
        raise ValueError(
            f"No stable Python reference root at P={pressure}, T={temperature}."
        )
    _, expected_volume, _ = min(stable)
    volume = eos.volume(pressure, temperature, params)
    np.testing.assert_allclose(volume, expected_volume, rtol=2e-12, atol=0.0)
    bulk_modulus = eos.isothermal_bulk_modulus_reuss(
        pressure, temperature, volume, params
    )
    values = (
        volume,
        bulk_modulus,
        eos.thermal_expansivity(pressure, temperature, volume, params),
        eos.gibbs_energy(pressure, temperature, volume, params),
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
            numbers += list(values[3:])
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
 * Absolute G, S and Cp use the shared HP2011 ideal-gas thermal reference.
 */
#pragma once
#include <array>

namespace ps1994_reference {{
struct State {{
  double pressure, temperature, volume, bulk_modulus, expansivity;
  double delta_gibbs, delta_entropy, delta_cp;
  double gibbs, entropy, cp;
  int stable_roots;
}};
inline constexpr double reference_pressure = {POLICY['reference_pressure_Pa']};
inline constexpr std::array<State, {len(rows)}> states = {{{{
  """ + ",\n  ".join(rows) + "\n}};\n} // namespace ps1994_reference\n"
    # Use the repository's C++ formatting when producing and checking snapshots.
    return format_header(text)


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
    else:
        try:
            check_snapshot(OUTPUT.read_text(), generated)
        except (ValueError, AssertionError) as error:
            parser.exit(
                1,
                f"PS1994 snapshots differ from Python BurnMan: {error}\n"
                "Regenerate with --write.\n",
            )
        print(f"Verified {len(STATES)} Python BurnMan PS1994 reference states.")


if __name__ == "__main__":
    main()
