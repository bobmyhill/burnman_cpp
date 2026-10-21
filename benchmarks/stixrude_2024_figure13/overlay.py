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
"""Overlay saved mode/iron/spin curves on the published Figure 13 vectors."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def _element(tag, **attrs):
    return ET.Element(
        f"{{{SVG}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items()}
    )


def _label(parent, x, y, content, size=9, anchor="middle"):
    e = _element(
        "text",
        x=x,
        y=y,
        font_size=size,
        text_anchor=anchor,
        font_family="Arial, Helvetica, sans-serif",
        fill="black",
    )
    e.text = content
    parent.append(e)


def _curve(x, y):
    """Emit separate subpaths around missing observations, preserving NaNs."""
    parts, continuing = [], False
    for xx, yy in zip(x, y):
        if not np.isfinite([xx, yy]).all():
            continuing = False
            continue
        parts.append(f"{'L' if continuing else 'M'}{xx:.5f},{yy:.5f}")
        continuing = True
    return " ".join(parts)


def make_overlay(grid_path, output_dir):
    """Register to each original panel's ticks; retain the original artwork.

    Calculated curves are not fitted, shifted or rescaled to improve agreement.
    Registration is affine in depth and fraction. The two published panels have
    different widths, so their tick calibrations are independent. Save vector
    SVG and high-resolution PNG; require rsvg-convert for the latter. No new
    equilibria are calculated.
    """
    grid_path, output_dir = Path(grid_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = HERE / "reference/figure13.svg"
    config = json.loads((HERE / "reference/axis_registration.json").read_text())
    original = ET.parse(reference).getroot()
    view_x, view_y, view_width, view_height = config["reference_viewbox"]
    centre = view_x + view_width / 2
    with np.load(grid_path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    depth = data["depth_km"]
    modes = list(data["cumulative_atomic_fractions"].T)
    panels = dict(
        modes=[*modes, 1 - 100 * data["native_iron_atomic_fraction"]],
        spin=[data["fp_low_spin_fraction"]],
    )
    output = _element(
        "svg",
        width=f"{view_width*2}pt",
        height="1010pt",
        viewBox=f"{view_x} 30 {view_width} 505",
        version="1.1",
    )
    defs = copy.deepcopy(original.find(f"{{{SVG}}}defs"))
    original_clip = _element("clipPath", id="benchmarkOriginalClip")
    original_clip.append(
        _element("rect", x=view_x, y=view_y, width=view_width, height=view_height)
    )
    defs.append(original_clip)
    output.append(defs)
    output.append(
        _element("rect", x=view_x, y=30, width=view_width, height=505, fill="white")
    )
    artwork = _element("g", clip_path="url(#benchmarkOriginalClip)")
    for child in original:
        if child.tag != f"{{{SVG}}}defs":
            artwork.append(copy.deepcopy(child))
    output.append(artwork)
    registration = {}
    ticks = np.asarray(config["depth_ticks_km"])
    for name, curves in panels.items():
        panel = config["panels"][name]
        centres = np.asarray(panel["tick_centres_pdf_points"])
        scale, origin = np.polyfit(ticks, centres, 1)
        left, top, right, bottom = panel["frame"]
        clip = _element("clipPath", id=f"benchmarkClip_{name}")
        clip.append(
            _element("rect", x=left, y=top, width=right - left, height=bottom - top)
        )
        defs.append(clip)
        overlay = _element(
            "g",
            clip_path=f"url(#benchmarkClip_{name})",
            fill="none",
            stroke="#005bff",
            stroke_width=0.65,
            stroke_linejoin="round",
            stroke_linecap="round",
        )
        for i, curve in enumerate(curves):
            options = dict(
                d=_curve(origin + scale * depth, bottom - (bottom - top) * curve)
            )
            if name == "modes" and i == len(curves) - 1:
                options.update(stroke_dasharray="3 2", stroke_width=0.9)
            if name == "spin":
                options.update(stroke_width=1.0)
            overlay.append(_element("path", **options))
        output.append(overlay)
        registration[name] = dict(
            x_scale_pdf_points_per_km=float(scale),
            x_origin_pdf_points=float(origin),
            frame_pdf_points=panel["frame"],
            maximum_tick_residual_km=float(
                np.max(np.abs(centres - origin - scale * ticks)) / scale
            ),
        )
    _label(
        output,
        centre,
        40,
        "Figure 13 benchmark: calculated curves over the published panels",
        9,
    )
    for x, text, dashed in [
        (95, "Published curves/data", False),
        (260, "Calculated modes / fp spin", False),
        (425, "Native Fe ×100", True),
    ]:
        color = "black" if x == 95 else "#005bff"
        opts = dict(x1=x - 20, y1=495, x2=x - 5, y2=495, stroke=color, stroke_width=1.0)
        if dashed:
            opts["stroke_dasharray"] = "3 2"
        output.append(_element("line", **opts))
        _label(output, x, 498, text, 7.5, anchor="start")
    _label(
        output,
        centre,
        514,
        "Native Fe dashed: 1 − 100 XFe. fp low spin: n(wuls) / [n(wu) + n(wuls)].",
        7,
    )
    _label(
        output,
        centre,
        527,
        "Original shading, ferric ratios, oxygen fugacity and external comparison data retained.",
        7,
    )
    svg = output_dir / "figure13_benchmark_overlay.svg"
    ET.ElementTree(output).write(svg, encoding="utf-8", xml_declaration=True)
    png = svg.with_suffix(".png")
    subprocess.run(
        ["rsvg-convert", "--width", "2400", "--output", str(png), str(svg)], check=True
    )
    metadata = dict(
        paper=config["paper"],
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        source_grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        registration="Affine calibration from panel ticks only; no fitting to calculated curves.",
        panels=registration,
        equilibrium_solves_performed=0,
    )
    (output_dir / "overlay_registration.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Overlay: {png}", flush=True)
    return svg, png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=HERE / "results/mode_box.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parents[1] / "build/benchmarks/stixrude_2024_figure13",
    )
    args = parser.parse_args()
    make_overlay(args.grid, args.output_dir)
