# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
"""Preserve the nine original liquid P-V/E-V panels and Table 1 reference.

Requires the supplied PDF and Poppler. Native EOSs are not imported. Tick
positions are snapped to the original vector artwork, independently of EOS
values. Settings and main(); no command-line arguments.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

PAPER_PDF = Path.home() / "deKoker_2013.pdf"
REFERENCE_DIR = Path(__file__).with_name("reference")
DPI = 250
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
# composition, PDF page, top/bottom row, temperatures, volume tick values and
# approximate positions (P then E), pressure ticks, energy ticks (kJ/mol).
SPECS = (
    (
        "MgO",
        17,
        "bottom",
        [2000, 3000, 4000, 5000, 6000, 7000, 10000],
        [9, 18],
        [180.8, 281.7],
        [316.6, 417.5],
        [0, 240],
        [581.8, 463.0],
        [-1200, -300],
        [586.5, 475.1],
    ),
    (
        "Mg5SiO7",
        18,
        "top",
        [2000, 3000, 4000, 6000, 8000],
        [60, 100],
        [180, 253.5],
        [315.9, 389.5],
        [0, 200],
        [294.2, 183.2],
        [-8000, -4000],
        [286.6, 191.4],
    ),
    (
        "Mg2SiO4",
        18,
        "bottom",
        [2000, 3000, 4000, 6000],
        [30, 60],
        [175.6, 264.0],
        [311.4, 399.8],
        [0, 200],
        [593.5, 481.0],
        [-4500, -3000],
        [578.5, 490.1],
    ),
    (
        "Mg3Si2O7",
        19,
        "top",
        [2000, 3000, 4000, 6000],
        [40, 120],
        [166.1, 282.0],
        [301.5, 417.4],
        [0, 240],
        [294.9, 187.9],
        [-8000, -5000],
        [285.0, 187.5],
    ),
    (
        "MgSiO3",
        19,
        "bottom",
        [2000, 3000, 4000, 6000, 8000],
        [20, 50],
        [175.8, 281.9],
        [311.8, 417.9],
        [0, 240],
        [595.0, 490.7],
        [-3500, -1500],
        [585.4, 475.3],
    ),
    (
        "MgSi2O5",
        20,
        "top",
        [2000, 3000, 4000, 6000, 8000],
        [30, 75],
        [177.3, 282.2],
        [312.7, 417.6],
        [0, 240],
        [293.9, 189.9],
        [-6000, -3000],
        [294.9, 192.0],
    ),
    (
        "MgSi3O7",
        20,
        "bottom",
        [2000, 3000, 4000, 6000, 8000],
        [30, 120],
        [158.5, 282.0],
        [294.9, 418.4],
        [0, 240],
        [596.8, 482.7],
        [-8000, -4000],
        [586.2, 483.3],
    ),
    (
        "MgSi5O11",
        21,
        "top",
        [2000, 3000, 4000, 5000, 6000, 8000],
        [60, 150],
        [164.6, 275.2],
        [300.3, 411.0],
        [0, 200],
        [294.4, 178.0],
        [-13000, -7000],
        [288.3, 183.8],
    ),
    (
        "SiO2",
        21,
        "bottom",
        [2000, 3000, 4000, 5000, 6000, 7000],
        [10, 30],
        [164.4, 282.0],
        [300.1, 417.7],
        [0, 200],
        [593.8, 488.7],
        [-2400, -1200],
        [587.4, 480.0],
    ),
)
TABLE_1 = {
    "volume_cm3_mol": (
        [16.46, 107.50, 58.40, 100.50, 41.8, 67.50, 93.50, 146.00, 27.8],
        [0.01, 0.01, 0.08, 0.08, 0.1, 0.09, 0.08, 0.01, 0.2],
    ),
    "K_T_GPa": (
        [33.49, 24.33, 19.50, 18.35, 17.2, 12.92, 12.53, 13.06, 6.2],
        [0.05, 0.02, 0.09, 0.04, 0.1, 0.06, 0.06, 0.05, 0.3],
    ),
    "entropy_J_mol_K": (
        [174.1, 1125.4, 595.1, 1010.8, 415.3, 641.1, 859.3, 1283.2, 205.5],
        [0.2, 0.5, 0.3, 0.5, 0.2, 0.5, 0.7, 1.7, 0.5],
    ),
    "Cv_per_nR": (
        [3.6, 3.9, 4.1, 4.2, 4.2, 4.5, 4.4, 4.5, 4.6],
        [0.2, 0.1, 0.4, 0.2, 0.4, 0.3, 0.3, 0.2, 0.2],
    ),
    "Cp_per_nR": (
        [4.5, 4.8, 4.4, 4.5, 4.4, 4.5, 4.6, 4.6, 4.6],
        [0.3, 0.2, 0.5, 0.3, 0.4, 0.4, 0.4, 0.3, 0.2],
    ),
    "grueneisen": (
        [0.87, 0.66, 0.31, 0.31, 0.27, 0.12, 0.18, 0.10, 0.06],
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
    ),
}


def straight_segments(svg):
    segments = []
    for element in ET.parse(svg).getroot().iter():
        a = element.attrib
        transform = [float(v) for v in re.findall(NUMBER, a.get("transform", ""))]
        if a.get("fill") != "none" or len(transform) != 6:
            continue
        aa, b, c, d, x, y = transform
        for path in re.findall(r"M[^M]*", a.get("d", "")):
            if re.search(r"[CQAZcqaz]", path):
                continue
            values = [float(v) for v in re.findall(NUMBER, path)]
            if len(values) != 4:
                continue
            points = np.array(values).reshape(2, 2) @ np.array([[aa, b], [c, d]]) + [
                x,
                y,
            ]
            segments.append((points, float(a["stroke-width"]) * abs(d)))
    return segments


def registration(
    segments,
    row,
    quantity,
    volume_values,
    volume_positions,
    values,
    positions,
    *,
    mgo=False,
):
    bottom = 590 if mgo else 600 if row == "bottom" else 300
    left, right = (150, 290) if quantity == "P" else (290, 425)
    horizontal = [
        (pts, w)
        for pts, w in segments
        if abs(pts[1, 1] - pts[0, 1]) < 0.01
        and abs(pts[1, 0] - pts[0, 0]) > 100
        and w > 0.7
        and left < pts[:, 0].min() < right
        and abs(pts[0, 1] - bottom) < 12
    ]
    if len(horizontal) != 1:
        raise ValueError(f"Ambiguous {quantity} axis: {horizontal}")
    axis = horizontal[0][0]
    x0, x1 = sorted(axis[:, 0])
    y1 = axis[0, 1]
    vertical = [
        pts
        for pts, w in segments
        if abs(pts[1, 0] - pts[0, 0]) < 0.01
        and abs(pts[1, 1] - pts[0, 1]) > 100
        and w > 0.7
        and abs(pts[0, 0] - x0) < 0.1
        and abs(pts[:, 1].max() - y1) < 0.1
    ]
    if len(vertical) != 1:
        raise ValueError(f"Ambiguous {quantity} vertical axis")
    y0 = vertical[0][:, 1].min()
    xticks = [
        pts[0, 0]
        for pts, w in segments
        if abs(pts[1, 0] - pts[0, 0]) < 0.01
        and 3 < abs(pts[1, 1] - pts[0, 1]) < 5.5
        and abs(pts[:, 1].min() - y1) < 0.1
    ]
    yticks = [
        pts[0, 1]
        for pts, w in segments
        if abs(pts[1, 1] - pts[0, 1]) < 0.01
        and 3 < abs(pts[1, 0] - pts[0, 0]) < 5.5
        and (
            abs(pts[:, 0].max() - x0) < 0.1
            if quantity == "P"
            else abs(pts[:, 0].min() - x1) < 0.1
        )
    ]

    def snap(position, candidates):
        value = min(candidates, key=lambda x: abs(x - position))
        if abs(value - position) > 6:
            raise ValueError(f"No tick near {position}: {candidates}")
        return float(value)

    return dict(
        bounds=[float(x0), float(y0), float(x1), float(y1)],
        volume_ticks_cm3_mol=[
            [v, snap(p, xticks)] for v, p in zip(volume_values, volume_positions)
        ],
        quantity_ticks=[[v, snap(p, yticks)] for v, p in zip(values, positions)],
    )


def main():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="dekoker_endmembers_") as directory:
        for i, (name, page, row, temperatures, v, xp, xe, p, yp, e, ye) in enumerate(
            SPECS
        ):
            stem = Path(directory) / f"page_{page}"
            if not stem.with_suffix(".svg").exists():
                subprocess.run(
                    [
                        "pdftocairo",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-svg",
                        str(PAPER_PDF),
                        str(stem.with_suffix(".svg")),
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "pdftoppm",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-r",
                        str(DPI),
                        "-singlefile",
                        "-png",
                        str(PAPER_PDF),
                        str(stem),
                    ],
                    check=True,
                    capture_output=True,
                )
            segments = straight_segments(stem.with_suffix(".svg"))
            panels = {
                q: registration(
                    segments, row, q, v, x, values, positions, mgo=name == "MgO"
                )
                for q, x, values, positions in (("P", xp, p, yp), ("E", xe, e, ye))
            }
            top, bottom = min(s["bounds"][1] for s in panels.values()), max(
                s["bounds"][3] for s in panels.values()
            )
            crop = [100, float(top - 9), 495, float(bottom + 36)]
            with Image.open(stem.with_suffix(".png")) as image:
                image.crop(tuple(round(c * DPI / 72) for c in crop)).save(
                    REFERENCE_DIR / f"{name}.png"
                )
            rows.append(
                dict(
                    composition=name,
                    pdf_page=page,
                    supplement_figure=f"S{i+2}",
                    temperatures_K=temperatures,
                    crop=crop,
                    panels=panels,
                    table_1={
                        key: dict(value=data[0][i], printed_uncertainty=data[1][i])
                        for key, data in TABLE_1.items()
                    },
                )
            )
    report = dict(
        source="de Koker, Karki & Stixrude (2013), doi:10.1016/j.epsl.2012.11.026",
        source_pdf_sha256=hashlib.sha256(PAPER_PDF.read_bytes()).hexdigest(),
        reference_state=dict(pressure_Pa=0, temperature_K=3000),
        units="Per mole of the named formula; volume cm3/mol, pressure GPa, energy kJ/mol; Cv/Cp divided by total atoms * R",
        compositions=rows,
    )
    (REFERENCE_DIR / "endmembers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(rows)} independent composition references")


if __name__ == "__main__":
    main()
