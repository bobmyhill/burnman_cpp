#!/usr/bin/env python3
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
"""Overlay saved Figure 14 velocities and phase boundaries on the original."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stixrude_2024 import CMB_DEPTH, PREM
from stixrude_2024_figure13.overlay import SVG, _element, _curve, _label


def compare_velocities(data, output_dir):
    """Quantify velocity differences against the original vector profiles."""
    import matplotlib.pyplot as plt

    root = ET.parse(HERE / "reference/figure14.svg").getroot()
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    colors = {
        "rgb(100%, 0%, 0%)": "pyrolite",
        "rgb(0%, 59.999084%, 0%)": "harzburgite",
        "rgb(0%, 24.313354%, 100%)": "basalt",
    }
    scale, origin = np.polyfit(
        [6.8, 7.0, 7.2, 7.4], [82.519531, 107.089844, 131.617188, 156.1875], 1
    )
    fig, ax = plt.subplots(figsize=(6, 7), layout="constrained")
    report = {}
    table = [
        "lithology,height_above_CMB_km,calculated_Vs_km_s,published_Vs_km_s,difference_km_s"
    ]
    for element in root.iter(f"{{{SVG}}}path"):
        name = colors.get(element.get("stroke"))
        if name is None:
            continue
        if set(re.findall(r"[A-DF-Za-df-z]", element.get("d"))) != {"M", "L"}:
            raise ValueError(
                "Reference velocity paths must contain only line segments."
            )
        xy = np.array([float(v) for v in re.findall(number, element.get("d"))]).reshape(
            -1, 2
        )
        a, b, c, d, tx, ty = map(float, re.findall(number, element.get("transform")))
        points = xy @ np.array([[a, b], [c, d]]) + np.array([tx, ty])
        reference_height = (186.886719 - points[:, 1]) * 600 / (186.886719 - 47.800781)
        reference_vs = (points[:, 0] - origin) / scale
        order = np.argsort(reference_height, kind="stable")
        h = data["profile_height_km"]
        ref = np.interp(h, reference_height[order], reference_vs[order])
        velocity = data[name + "_vs"] / 1000
        difference = velocity - ref
        report[name] = dict(
            rms_difference_km_s=float(np.sqrt(np.mean(difference**2))),
            maximum_absolute_difference_km_s=float(np.max(abs(difference))),
        )
        table.extend(
            f"{name},{hh:.9g},{v:.12g},{r:.12g},{dv:.12g}"
            for hh, v, r, dv in zip(h, velocity, ref, difference)
        )
        ax.plot(
            difference,
            h,
            label=name.capitalize(),
            color=dict(pyrolite="red", harzburgite="#009900", basalt="#003eff")[name],
        )
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set(
        xlabel="Calculated minus published Vs (km s⁻¹)",
        ylabel="Height above CMB (km)",
        ylim=(0, 600),
        title="Figure 14 velocity residuals",
    )
    ax.legend()
    output_dir = Path(output_dir)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"velocity_residuals.{extension}", dpi=200)
    plt.close(fig)
    (output_dir / "velocity_comparison.csv").write_text("\n".join(table) + "\n")
    (output_dir / "velocity_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def make_overlay(grid_path, diagram_path, output_dir):
    """Use independent panel ticks; do not fit calculated curves to the paper.

    Original solid velocity profiles retain their colours; calculated profiles
    are dashed in matching colours. Cyan lines are the new phase boundaries.
    The original expansivity map remains visible; its fully recalculated
    counterpart is saved separately as figure14.png/svg.
    """
    grid_path, diagram_path, output_dir = map(
        Path, (grid_path, diagram_path, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = HERE / "reference/figure14.svg"
    original = ET.parse(reference).getroot()
    with np.load(grid_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    saved = json.loads(diagram_path.read_text())
    diagram = saved.get("result", saved)
    output = _element(
        "svg", width="1410pt", height="1188pt", viewBox="55 30 235 198", version="1.1"
    )
    defs = copy.deepcopy(original.find(f"{{{SVG}}}defs"))
    clip = _element("clipPath", id="figure14ArtworkClip")
    clip.append(_element("rect", x=55, y=40, width=235, height=170))
    defs.append(clip)
    output.append(defs)
    output.append(_element("rect", x=55, y=30, width=235, height=198, fill="white"))
    artwork = _element("g", clip_path="url(#figure14ArtworkClip)")
    for child in original:
        if child.tag != f"{{{SVG}}}defs":
            artwork.append(copy.deepcopy(child))
    output.append(artwork)
    for name, frame in dict(
        velocity=[82.519531, 47.800781, 168.453125, 186.886719],
        thermal=[182.170823, 47.496114, 268.296883, 186.886113],
    ).items():
        left, top, right, bottom = frame
        panelclip = _element("clipPath", id=f"figure14_{name}")
        panelclip.append(
            _element("rect", x=left, y=top, width=right - left, height=bottom - top)
        )
        defs.append(panelclip)
        group = _element(
            "g",
            clip_path=f"url(#figure14_{name})",
            fill="none",
            stroke_linejoin="round",
            stroke_linecap="round",
        )
        if name == "velocity":
            # 6.8, 7.0, 7.2, 7.4 km/s ticks of the left panel.
            scale, origin = np.polyfit(
                [6.8, 7.0, 7.2, 7.4], [82.519531, 107.089844, 131.617188, 156.1875], 1
            )
            y = bottom - data["profile_height_km"] * (bottom - top) / 600
            for lithology, color in dict(
                pyrolite="red", harzburgite="#009900", basalt="#003eff"
            ).items():
                group.append(
                    _element(
                        "path",
                        d=_curve(origin + scale * data[lithology + "_vs"] / 1000, y),
                        stroke=color,
                        stroke_width=0.8,
                        stroke_dasharray="2 1.2",
                    )
                )
        else:
            xscale = 3000 / (251.025847 - 186.566462)

            def project(temperature, height):
                return (
                    186.566462 + (np.asarray(temperature) - 1000) / xscale,
                    bottom - np.asarray(height) * (bottom - top) / 600,
                )

            for boundary in diagram["boundaries"]:
                pressure = [p["pressure"] for p in boundary["points"]]
                temperature = [p["temperature"] for p in boundary["points"]]
                height = (CMB_DEPTH - PREM.depth(pressure)) / 1000
                group.append(
                    _element(
                        "path",
                        d=_curve(*project(temperature, height)),
                        stroke="#00e2ee",
                        stroke_width=0.5,
                    )
                )
            group.append(
                _element(
                    "path",
                    d=_curve(
                        *project(
                            data["geotherm_temperature"], data["geotherm_height_km"]
                        )
                    ),
                    stroke="#009900",
                    stroke_width=0.8,
                    stroke_dasharray="2 1.2",
                )
            )
        output.append(group)
    _label(
        output,
        172.5,
        36,
        "Figure 14: calculated curves over the published panels",
        size=6,
    )
    _label(
        output,
        172.5,
        217,
        "Original: solid curves. Calculated: dashed Vs and geotherm; cyan phase boundaries.",
        size=4.7,
    )
    _label(
        output,
        172.5,
        224,
        "Reference colours show published alpha; see figure14.png for recalculated alpha.",
        size=4.7,
    )
    svg = output_dir / "figure14_benchmark_overlay.svg"
    ET.ElementTree(output).write(svg, encoding="utf-8", xml_declaration=True)
    png = svg.with_suffix(".png")
    subprocess.run(
        ["rsvg-convert", "--width", "2400", "--output", str(png), str(svg)], check=True
    )
    (output_dir / "overlay_registration.json").write_text(
        json.dumps(
            dict(
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                source_grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                source_diagram_sha256=hashlib.sha256(
                    diagram_path.read_bytes()
                ).hexdigest(),
                registration="Independent affine panel-tick calibration; no fitting to calculated curves.",
                velocity_ticks_km_per_s=[6.8, 7.0, 7.2, 7.4],
                velocity_tick_positions_pdf_points=[
                    82.519531,
                    107.089844,
                    131.617188,
                    156.1875,
                ],
                velocity_height_frame_pdf_points=[47.800781, 186.886719],
                thermal_height_frame_pdf_points=[47.496114, 186.886113],
                temperature_ticks_K=[1000, 4000],
                temperature_ticks_pdf_points=[186.566462, 251.025847],
                equilibrium_solves_performed=0,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Overlay: {png}", flush=True)
    compare_velocities(data, output_dir)
    return svg, png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parents[1] / "build/benchmarks/stixrude_2024_figure14",
    )
    parser.add_argument("--grid", type=Path, default=HERE / "results/figure14.npz")
    parser.add_argument(
        "--diagram", type=Path, default=HERE / "results/phase_diagram.json"
    )
    args = parser.parse_args()
    make_overlay(args.grid, args.diagram, args.output_dir)
