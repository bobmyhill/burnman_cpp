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

"""
This example is a follow-up to example_pyrolite_fugacity_comparison.py.
It reads the saved equilibrium grids for pyrolites with
initial Fe3+/total Fe of 3% and 10% (on a metal-free basis).
It calculates the oxygen fugacity difference
and the offset from a fayalite-magnetite-quartz buffer.

At common P,T, delta log10(fO2) = (mu_{O2,10%} - mu_{O2,3%})/(R*T*ln(10)),
such that a positive difference means the 10% bulk has higher fO2.
Contours use only cells valid in both compositions.

The fayalite-magnetite-quartz buffer is taken from
Stixrude and Lithgow-Bertelloni (2024) and calculated as:
mu(O2) = 2 G_mt + 3 G_qtz - 3 G_fa.
This is a metastable assemblage at 20-25 GPa.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024, load_dataset, model_sets
from burnman_cpp.tools.phase_diagram_helpers import save_figure

SOURCE = Path(__file__).resolve().with_name("pyrolite_fugacity_comparison_output")
OUTPUT_DIR = SOURCE / "fugacity_difference"
COMPOSITIONS = ("0.03", "0.10")  # Difference is second minus first.
CONTOUR_LEVELS = 12  # An integer chooses levels automatically; a sequence fixes them.


def oxygen_potential_grid(grid, phases=None):
    """Read mu(O2) in J/mol from the exact saved active solution bases."""
    if phases is not None:
        grid.bind_phases(phases)
    if not grid.has_solutions:
        raise ValueError("The grid must retain its equilibrium states.")
    values = np.full(grid.equilibrium_success.shape, np.nan)
    diagnostics = []
    for j, i in np.argwhere(grid.equilibrium_success):
        saved = grid.solution_at(int(j), int(i))
        try:
            assemblage = bm.reconstruct_assemblage(saved)
            values[j, i] = assemblage.chemical_potential([{"O": 2.0}])[0]
        except (ValueError, RuntimeError) as error:
            diagnostics.append(f"grid [{j}, {i}]: {error}")
    return values, diagnostics


def log_fugacity_difference(reference, comparison, phases=None):
    """Return delta log10(fO2), both mu(O2) arrays and evaluation diagnostics."""
    if reference.diagram_type != "PT" or comparison.diagram_type != "PT":
        raise ValueError("The comparison requires two P-T grids.")
    if not (
        np.array_equal(reference.x, comparison.x)
        and np.array_equal(reference.y, comparison.y)
    ):
        raise ValueError("Both compositions must use exactly the same P-T grid.")
    reference_grid, reference_diagnostics = oxygen_potential_grid(reference, phases)
    comparison_grid, comparison_diagnostics = oxygen_potential_grid(comparison, phases)
    difference = (comparison_grid - reference_grid) / (
        bm.constants.gas_constant * reference.y[:, None] * np.log(10.0)
    )
    return (
        difference,
        (reference_grid, comparison_grid),
        {
            "reference": reference_diagnostics,
            "comparison": comparison_diagnostics,
        },
    )


def fmq_potential_grid(grid):
    """O2 chemical potential of the fixed, metastable SLB2024 FMQ reference."""
    buffer = bm.CombinedMineral(
        [SLB_2024.fayalite(), SLB_2024.magnetite(), SLB_2024.quartz()],
        [-3.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        name="FMQ",
    )
    pressure, temperature = np.meshgrid(grid.x, grid.y)
    return buffer.evaluate(["molar_gibbs"], pressure, temperature)[0]


def plot_contours(ax, grid, values, title, filled_levels=21):
    temperature, pressure = np.meshgrid(grid.y - 273.15, grid.x / 1.0e9, indexing="ij")
    masked = np.ma.masked_invalid(values)
    filled = ax.contourf(
        temperature,
        pressure,
        masked,
        levels=filled_levels,
        cmap="viridis",
        corner_mask=False,
    )
    lines = ax.contour(
        temperature,
        pressure,
        masked,
        levels=CONTOUR_LEVELS,
        colors="black",
        linewidths=0.7,
        corner_mask=False,
    )
    ax.clabel(lines, fmt="%.2f", fontsize=7)
    ax.set(title=title, xlabel="Temperature (°C)")
    return filled


def main():
    diagrams = [
        bm.load_phase_diagram(SOURCE / name / "refined.json") for name in COMPOSITIONS
    ]
    models = model_sets.slb2024()
    provenance = diagrams[0].get("workflow", {}).get("models", {})
    dataset = models if provenance.get("name") == models.name else SLB_2024
    phases, _ = load_dataset(dataset)
    grids = [
        bm.load_property_grid(SOURCE / name / "grid_equilibria.json.gz", phases=phases)
        for name in COMPOSITIONS
    ]
    difference, potentials, diagnostics = log_fugacity_difference(*grids)
    fmq = fmq_potential_grid(grids[0])
    scale = bm.constants.gas_constant * grids[0].y[:, None] * np.log(10.0)
    offsets = [(mu - fmq) / scale for mu in potentials]
    valid = np.isfinite(difference)
    if not valid.any():
        raise RuntimeError(
            "No saved equilibrium pair defines an oxygen fugacity difference."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "fugacity_difference.npz",
        pressure_Pa=grids[0].x,
        temperature_K=grids[0].y,
        mu_O2_reference_J_mol=potentials[0],
        mu_O2_comparison_J_mol=potentials[1],
        delta_log10_fO2=difference,
        mu_O2_FMQ_J_mol=fmq,
        delta_FMQ_reference=offsets[0],
        delta_FMQ_comparison=offsets[1],
        valid=valid,
    )
    (OUTPUT_DIR / "diagnostics.json").write_text(
        json.dumps(
            dict(
                reference=COMPOSITIONS[0],
                comparison=COMPOSITIONS[1],
                equilibrium_solves=0,
                valid_points=int(valid.sum()),
                total_points=int(valid.size),
                fmq_reference="SLB2024: fixed fayalite, magnetite, quartz (metastable)",
                diagnostics=diagnostics,
            ),
            indent=2,
        )
        + "\n"
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(19, 6), sharex=True, sharey=True, layout="constrained"
    )
    for ax, diagram, name, title in zip(
        axes[:2],
        diagrams,
        COMPOSITIONS,
        ("Initial Fe³⁺/ΣFe = 3%", "Initial Fe³⁺/ΣFe = 10%"),
    ):
        bm.plot_phase_diagram(
            diagram,
            ax,
            pressure_unit="GPa",
            temperature_unit="C",
            phase_aliases=SLB_2024.phase_aliases,
            label_fontsize=6,
            colorbar=False,
            label_key_path=OUTPUT_DIR / f"{name}_assemblages.md",
        )
        ax.set_title(title)
    filled = plot_contours(axes[2], grids[0], difference, "Δlog₁₀ fO₂: 10% - 3%")
    figure.colorbar(filled, ax=axes[2], label="Δlog₁₀ fO₂")
    for ax in axes[1:]:
        ax.set_ylabel("")
    save_figure(figure, OUTPUT_DIR / "fugacity_difference.png", formats=("svg",))
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, sharey=True, layout="constrained"
    )
    levels = np.linspace(np.nanmin(offsets), np.nanmax(offsets), 21)
    for ax, values, label in zip(axes, offsets, ("3%", "10%")):
        filled = plot_contours(
            ax, grids[0], values, f"Initial Fe³⁺/ΣFe = {label}", levels
        )
    axes[0].set_ylabel("Pressure (GPa)")
    figure.colorbar(filled, ax=list(axes), label="ΔFMQ = log₁₀(fO₂ / fO₂,FMQ)")
    figure.suptitle("SLB2024. Fixed fayalite-magnetite-quartz reference")
    save_figure(figure, OUTPUT_DIR / "fmq_offsets.png", formats=("svg",))
    print(
        f"Δlog10(fO2), 10% minus 3%: {difference[valid].min():.6g} to {difference[valid].max():.6g}"
    )
    print(f"Figure and numerical arrays: {OUTPUT_DIR}")
    for name, values in zip(COMPOSITIONS, offsets):
        print(f"{name}: ΔFMQ {np.nanmin(values):.6g} to {np.nanmax(values):.6g}")


if __name__ == "__main__":
    main()
