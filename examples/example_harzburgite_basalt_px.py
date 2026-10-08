#!/usr/bin/env python3
"""Harzburgite–basalt P-X section, SLB2024 solids, 1673 K and 0–25 GPa.

Table 1 of Stixrude & Lithgow-Bertelloni (2024), doi:10.1093/gji/ggae126,
supplies both dry oxide compositions, including their different ferric ratios.
Each native Composition is normalized to 100 g (0.1 kg). X is the basalt mass
fraction: bulk(X)=(1-X)*harzburgite+X*basalt, always representing 100 g.
The pyrolite example supplies the complete SLB2024 candidate set and settings.
All stability searches, equilibrium and tracing run in C++; Python plots.
The pyrolite EOS-domain checks are retained. Melting is excluded; oxygen is
closed and unbuffered.

Run: python examples/example_harzburgite_basalt_px.py
"""

import argparse
from pathlib import Path

import burnman_cpp as bm
from example_pyrolite_pseudosection import (
    PAPER_URL,
    candidate_phases,
    settings,
)


def endmember(oxides):
    composition = bm.Composition(oxides, unit_type="mass")
    composition.renormalize("mass", "total", 0.1)
    return composition


HARZBURGITE = endmember(
    dict(
        SiO2=43.43,
        MgO=45.93,
        FeO=8.34,
        Fe2O3=0.09,
        CaO=0.90,
        Al2O3=1.00,
        Na2O=0.01,
        Cr2O3=0.30,
    )
)
BASALT = endmember(
    dict(
        SiO2=50.42,
        MgO=9.77,
        FeO=7.10,
        Fe2O3=1.07,
        CaO=12.54,
        Al2O3=16.80,
        Na2O=2.23,
        Cr2O3=0.07,
    )
)


def calculate(seeds=7, quick=False, verbose=False):
    opts = settings(seeds, quick, verbose)
    opts.composition_seeds = opts.pressure_seeds
    return bm.pseudosection(
        HARZBURGITE.atomic_composition,
        candidate_phases(),
        diagram="PX",
        composition_end=BASALT.atomic_composition,
        pressure_range=(0.0, 25.0e9),
        temperature=1673.0,
        settings=opts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-json", type=Path, help="Replot saved results.")
    parser.add_argument("--label-fontsize", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "harzburgite_basalt_px_output",
    )
    args = parser.parse_args()
    saved = bm.load_pseudosection(args.plot_json) if args.plot_json else {}
    result = (
        bm.PseudosectionResult.from_dict(saved)
        if args.plot_json
        else calculate(args.seeds, args.quick, args.verbose)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "harzburgite_basalt_px"
    data = saved | result.to_dict()
    data.update(
        endpoint_masses_kg={
            "harzburgite": HARZBURGITE.mass_composition,
            "basalt": BASALT.mass_composition,
        },
        composition_source=PAPER_URL,
        closed_system=True,
    )
    bm.save_pseudosection(data, stem.with_suffix(".json"))
    if not args.no_plots:
        fig, ax = bm.plot_pseudosection(
            result,
            pressure_unit="GPa",
            composition_label="Basalt mass fraction",
            label_fontsize=args.label_fontsize,
            label_key_path=stem.with_name(stem.name + "_labels.md"),
        )
        ax.set_title("Harzburgite–basalt at 1673 K (SLB2024 solids)")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"{len(result.boundaries)} phase boundaries; resolved={result.resolved}")
    for diagnostic in result.diagnostics:
        print(diagnostic)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
