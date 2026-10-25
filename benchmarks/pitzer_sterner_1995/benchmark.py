# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Native H2O/CO2 compression factors against Pitzer & Sterner (1995), Fig. 1.

Run: python benchmarks/pitzer_sterner_1995/benchmark.py
Requires NumPy and Matplotlib; neither Python BurnMan nor the PDF is needed.
"""

import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import burnman_cpp as bm

REFERENCE_DIR = Path(__file__).with_name("reference")
OUTPUT_DIR = Path(__file__).with_name("results")


def parameters(fluid):
    """Table I mechanics with an arbitrary ideal-gas caloric reference.

    Figure 1 does not constrain the density-independent thermal function.
    Its arbitrary zero and constant Cp do not affect P, K_T, Z or V(P,T).
    """
    return bm.mineral_params(
        dict(
            equation_of_state="pitzer-sterner",
            formula=fluid["formula"],
            n=sum(fluid["formula"].values()),
            molar_mass=fluid["molar_mass_kg_mol"],
            c_coeffs=fluid["c_coeffs"],
            H_0=0.0,
            S_0=0.0,
            Cp=[3.5 * bm.constants.gas_constant, 0.0, 0.0, 0.0],
        )
    )


def pixel_values(pixels, axes):
    left, top, right, bottom = axes["bounds_pixels"]
    xy = np.asarray(pixels)
    return (xy[:, 0] - left) * 7.0 / (right - left), (bottom - xy[:, 1]) * 25.0 / (
        bottom - top
    )


def compression_factor(eos, params, t, density, critical_density):
    # Molar density is in mol/m3 here; c_coeffs use mol/cm3 internally.
    rho = np.asarray(density) * critical_density
    pressures = np.array([eos.compute_pressure(t, 1.0 / r, params) for r in rho])
    return pressures / (rho * bm.constants.gas_constant * t), pressures


def main():
    reference = json.loads((REFERENCE_DIR / "figure1.json").read_text())
    t = reference["temperature_K"]
    axes = reference["axes"]
    eos = bm.make_eos("pitzer-sterner")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 6.9), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    overlay = fig.add_subplot(grid[0, 0])
    residual = fig.add_subplot(grid[0, 1])
    image = mpimg.imread(REFERENCE_DIR / "figure1.png")
    left, top, right, bottom = axes["bounds_pixels"]
    # Register the published scan using its axes, without fitting the EOS.
    dx, dy = (right - left) / 7.0, (bottom - top) / 25.0
    overlay.imshow(
        image,
        extent=[
            -left / dx,
            (image.shape[1] - left) / dx,
            (bottom - image.shape[0]) / dy,
            bottom / dy,
        ],
        aspect="auto",
        alpha=0.6,
    )
    overlay.set(xlim=(-2.1, 7.5), ylim=(-6.0, 26.5))
    overlay.set_title("Figure 1: 1600 K", pad=24)
    overlay.set_axis_off()
    residual.axhline(0.0, color="0.6", lw=0.8)
    rows, isotherms, summary = [], [], {}
    for (name, fluid), color in zip(
        reference["fluids"].items(), ["#0072B2", "#D55E00"]
    ):
        params = parameters(fluid)
        critical = fluid["critical_molar_density_mol_m3"]
        x = np.linspace(1.0e-5, 7.0, 701)
        z, pressure = compression_factor(eos, params, t, x, critical)
        visible = z <= 25.0
        overlay.plot(
            x[visible], z[visible], color=color, ls="--", lw=1.5, label=f"Native {name}"
        )
        isotherms.extend(zip([name] * len(x), x, pressure, z))
        xr, zr = pixel_values(fluid["curve_pixels"], axes)
        native, pr = compression_factor(eos, params, t, xr, critical)
        delta = native - zr
        # Image reading uncertainty in both coordinates, propagated to Z.
        px = reference["digitization"]["coordinate_uncertainty_pixels"]
        xerr = px / dx
        upper, _ = compression_factor(eos, params, t, xr + xerr, critical)
        lower, _ = compression_factor(eos, params, t, xr - xerr, critical)
        uncertainty = px / dy + np.maximum(abs(upper - native), abs(lower - native))
        residual.errorbar(
            xr,
            delta,
            xerr=xerr,
            yerr=uncertainty,
            fmt="o",
            ms=4,
            color=color,
            label=name,
            capsize=2,
        )
        summary[name] = dict(
            count=len(xr),
            rms_delta_Z=float(np.sqrt(np.mean(delta**2))),
            max_abs_delta_Z=float(np.max(abs(delta))),
            rms_relative_difference=float(np.sqrt(np.mean((delta / zr) ** 2))),
        )
        rows.extend(zip([name] * len(xr), xr, zr, native, delta, uncertainty, pr))
        # The benchmark also exercises density inversion at printed curve points.
        volumes = np.array([eos.compute_volume(float(p), t, params) for p in pr])
        np.testing.assert_allclose(volumes, 1.0 / (xr * critical), rtol=1e-11)
    overlay.legend(
        loc="lower center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    residual.set(
        xlabel=r"Reduced density $\rho/\rho_c$",
        ylabel=r"Native $Z$ − digitized $Z$",
    )
    residual.legend(frameon=False)
    residual.grid(alpha=0.2)
    residual.set_title("Published fitted curves")
    fig.suptitle("Pitzer & Sterner (1995): native H₂O and CO₂ EOS", fontsize=14)
    fig.supxlabel(
        "Error bars: reading bounds from ±2 pixels in x and y.",
        fontsize=8,
    )
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    for filename, header, data in [
        (
            "digitized_comparison.csv",
            [
                "fluid",
                "reduced_density",
                "paper_Z",
                "native_Z",
                "delta_Z",
                "reading_bound_Z",
                "pressure_Pa",
            ],
            rows,
        ),
        ("isotherms.csv", ["fluid", "reduced_density", "pressure_Pa", "Z"], isotherms),
    ]:
        with (OUTPUT_DIR / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(data)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT_DIR / 'comparison.pdf'}")


if __name__ == "__main__":
    main()
