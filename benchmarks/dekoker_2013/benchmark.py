# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
"""Compare the nine native DKS liquid EOSs with the original paper.

Only individual-composition P(V,T), E(V,T) and Table 1 properties are compared.
No mixing projection, entropy reset, solid model or melting calculation.
Run: python benchmarks/dekoker_2013/benchmark.py
Settings and main(); no command-line arguments.
"""

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

import burnman_cpp as bm

REFERENCE_DIR = Path(__file__).with_name("reference")
OUTPUT_DIR = Path(__file__).with_name("results")
SHOW_PLOTS = False
VOLUME_SAMPLES = 401


def load_reference():
    return json.loads((REFERENCE_DIR / "endmembers.json").read_text())


def native_table_1(phase):
    phase.set_state(0.0, 3000.0)
    nR = sum(phase.formula.values()) * bm.constants.gas_constant
    return dict(
        volume_cm3_mol=phase.molar_volume * 1.0e6,
        K_T_GPa=phase.isothermal_bulk_modulus_reuss * 1.0e-9,
        entropy_J_mol_K=phase.molar_entropy,
        Cv_per_nR=phase.molar_heat_capacity_v / nR,
        Cp_per_nR=phase.molar_heat_capacity_p / nR,
        grueneisen=phase.grueneisen_parameter,
    )


def axis_value(coordinate, ticks):
    (v0, c0), (v1, c1) = ticks
    return v0 + (coordinate - c0) * (v1 - v0) / (c1 - c0)


def internal_energy_at_volumes(phase, temperature, volumes, pressures):
    """Read native energies only where V(P(V,T),T) returns the same branch."""
    energies = np.full(len(volumes), np.nan)
    for i, (volume, pressure) in enumerate(zip(volumes, pressures)):
        try:
            phase.set_state(pressure, temperature)
            if np.isclose(phase.molar_volume, volume, rtol=1e-10, atol=1e-16):
                energies[i] = phase.molar_internal_energy / 1e3
        except (ValueError, RuntimeError):
            continue
    return energies


def comparison_figure(phase, reference):
    x0, y0, x1, y1 = reference["crop"]
    width = 10.0
    image_height = width * (y1 - y0) / (x1 - x0)
    fig = plt.figure(figsize=(width, image_height + 0.9))
    fraction = image_height / fig.get_figheight()
    bg = fig.add_axes([0, 0, 1, fraction])
    bg.imshow(
        mpimg.imread(REFERENCE_DIR / f"{reference['composition']}.png"), alpha=0.5
    )
    bg.set_axis_off()
    fig.text(
        0.5,
        0.98,
        f"{reference['composition']} liquid: native DKS EOS / Figure {reference['supplement_figure']}",
        ha="center",
        va="top",
        fontsize=13,
    )
    handles = [
        Line2D([], [], color=".6", label="Original LFTR fits and FPMD points (faint)")
    ]
    temperatures = reference["temperatures_K"]
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, len(temperatures)))
    handles += [
        Line2D([], [], color=c, ls="--", label=f"{t:g} K")
        for t, c in zip(temperatures, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.32 / fig.get_figheight()),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    eos = bm.make_eos("dks_l")
    params = phase.params
    rows = []
    for quantity, spec in reference["panels"].items():
        left, top, right, bottom = spec["bounds"]
        ax = fig.add_axes(
            [
                (left - x0) / (x1 - x0),
                (y1 - bottom) / (y1 - y0) * fraction,
                (right - left) / (x1 - x0),
                (bottom - top) / (y1 - y0) * fraction,
            ],
            frameon=False,
        )
        ax.set(
            xlim=[axis_value(x, spec["volume_ticks_cm3_mol"]) for x in (left, right)],
            ylim=[axis_value(y, spec["quantity_ticks"]) for y in (bottom, top)],
        )
        ax.set_axis_off()
        volumes = np.linspace(*ax.get_xlim(), VOLUME_SAMPLES) * 1.0e-6
        for temperature, color in zip(temperatures, colors):
            pressures = [
                eos.compute_pressure(temperature, float(v), params) for v in volumes
            ]
            values = (
                np.array(pressures) / 1.0e9
                if quantity == "P"
                else internal_energy_at_volumes(phase, temperature, volumes, pressures)
            )
            if np.count_nonzero(np.isfinite(values)) < 3:
                raise RuntimeError(
                    f"No stable {quantity} curve for {phase.name} at {temperature} K"
                )
            ax.plot(volumes * 1e6, values, color=color, ls="--", lw=1.1)
            rows.extend(
                [
                    reference["composition"],
                    temperature,
                    quantity,
                    float(v),
                    float(value),
                ]
                for v, value in zip(volumes, values)
                if np.isfinite(value)
            )
    return fig, rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    results, isotherms, table_rows = [], [], []
    with PdfPages(OUTPUT_DIR / "endmember_comparisons.pdf") as pdf:
        for row in reference["compositions"]:
            name = row["composition"]
            phase = getattr(bm.minerals.DKS_2013_liquids, name + "_liquid")()
            calculated = native_table_1(phase)
            comparison = {}
            for prop, native in calculated.items():
                source = row["table_1"][prop]
                delta = native - source["value"]
                comparison[prop] = dict(
                    **source,
                    native=native,
                    native_minus_paper=delta,
                    difference_in_printed_uncertainties=delta
                    / source["printed_uncertainty"],
                )
                table_rows.append(
                    [
                        name,
                        prop,
                        source["value"],
                        native,
                        delta,
                        source["printed_uncertainty"],
                    ]
                )
            results.append(dict(composition=name, properties=comparison))
            fig, rows = comparison_figure(phase, row)
            pdf.savefig(fig)
            for extension in ("pdf", "png"):
                fig.savefig(OUTPUT_DIR / f"{name}_comparison.{extension}", dpi=200)
            isotherms.extend(rows)
            if not SHOW_PLOTS:
                plt.close(fig)
            print(
                f"Compared {name}: Figure {row['supplement_figure']} and Table 1",
                flush=True,
            )
    for filename, fields, rows in (
        (
            "table_1_comparison.csv",
            [
                "composition",
                "property",
                "paper",
                "native",
                "native_minus_paper",
                "printed_uncertainty",
            ],
            table_rows,
        ),
        (
            "isotherms.csv",
            [
                "composition",
                "temperature_K",
                "quantity",
                "volume_m3_mol",
                "value_GPa_or_kJ_mol",
            ],
            isotherms,
        ),
    ):
        with (OUTPUT_DIR / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    report = dict(
        source=reference["source"],
        source_pdf_sha256=reference["source_pdf_sha256"],
        burnman_cpp_version=bm.__version__,
        scope="Nine independent liquid compositions only; no continuous mixing or phase diagram",
        comparisons=results,
    )
    (OUTPUT_DIR / "endmember_comparisons.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
