#!/usr/bin/env python3
"""Dry closed-system pyrolite, SLB2024, 0–25 GPa and 0–2500 K.

All thermodynamics, polytope construction, Gibbs minimisation, equilibrium
solves and phase-line continuation run in C++; Python writes JSON and plots.
The composition follows Table 1 of Stixrude & Lithgow-Bertelloni (2024),
doi:10.1093/gji/ggae126 (Workman & Hart 2005 pyrolite). Total Fe is conserved
while the starting molar Fe3+/sum(Fe) ratio is changed from 2% to exactly 3%.
Oxygen is closed and unbuffered. All three native iron polymorphs are allowed,
as are separate high/low spin endmembers within ferropericlase and bridgmanite.
Their spin populations equilibrate in the same C++ solve as the compositions.

SLB2024 contains solids, not silicate or iron liquids. This is a subsolidus
model extrapolated across the requested window, not a melting phase diagram.
At low P / high T some candidate EOS have no mechanically stable volume.
Only diagnosed EOS-domain failures may be excluded, with the reasons recorded
for every sample in JSON; numerical failures remain failures. The Mg-Tschermak
thermal spinodal makes the full orthopyroxene model undefined at LP/HT; this
example conservatively masks that domain even when other phases remain valid.
A solid outside its EOS domain is never assigned an unstable volume root.
Zero kelvin is included using active solution faces. Both the magnetic and
Debye contributions have finite analytic zero-temperature limits. All stable
states are verified against the full solution composition polytopes.

Increase --seeds to check narrow/disconnected fields. --refine-json resumes
unfinished lines from the last accepted native phase compositions and amounts.
"""

import argparse
import json
from math import isclose
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24

PAPER_URL = "https://academic.oup.com/gji/article/237/3/1699/7640845"
INITIAL_FERRIC_FRACTION = 0.03

P_Pa = 25.0e9
T_K = 2500.0


def pyrolite_composition(ferric_fraction=INITIAL_FERRIC_FRACTION):
    """Table 1 bulk with unchanged total Fe and a chosen molar ferric ratio.

    The tabulated masses are relative gram amounts (their rounded sum is not
    exactly 100). Normalize only after redistributing iron and adding oxygen.
    Return a native Composition representing 100 g (0.1 kg) of rock. All
    formula conversions and composition arithmetic run in C++.
    """
    if not 0.0 <= ferric_fraction <= 1.0:
        raise ValueError("ferric_fraction must be between zero and one.")
    iron = bm.Composition(dict(FeO=8.05e-3, Fe2O3=0.18e-3), unit_type="mass")
    total_fe = iron.atomic_composition["Fe"]
    composition = bm.Composition(
        dict(
            SiO2=45.0e-3,
            MgO=38.88e-3,
            CaO=3.18e-3,
            Al2O3=4.0e-3,
            Na2O=0.13e-3,
            Cr2O3=0.57e-3,
        ),
        unit_type="mass",
    )
    # FeO contains one Fe atom; Fe2O3 contains two. The initial oxygen
    # inventory follows the chosen ferric ratio and is closed during solving.
    composition.add_components(
        dict(
            FeO=total_fe * (1.0 - ferric_fraction),
            Fe2O3=total_fe * ferric_fraction / 2.0,
        ),
        unit_type="molar",
    )
    composition.renormalize("mass", "total", 0.1)
    return composition


PYROLITE_COMPOSITION = pyrolite_composition()
# Masses in grams of a 100 g rock are also oxide weight percentages for JSON.
PYROLITE_OXIDES = {
    oxide: kg * 1.0e3 for oxide, kg in PYROLITE_COMPOSITION.mass_composition.items()
}


PHASE_GLOSSARY = {
    "ol": "Olivine",
    "wa": "Wadsleyite",
    "ri": "Ringwoodite",
    "opx": "Orthopyroxene",
    "cpx": "Clinopyroxene",
    "hpcpx": "High-pressure C2/c pyroxene",
    "gt": "Garnet / majorite",
    "sp": "Spinel",
    "pl": "Plagioclase",
    "ak": "Akimotoite / corundum / hematite solution",
    "fp": "Ferropericlase, including high- and low-spin iron",
    "bg": "Bridgmanite, including high- and low-spin ferric iron",
    "ppv": "Post-perovskite",
    "cf": "Calcium-ferrite-structured solution",
    "nal": "New aluminous phase",
    "capv": "CaSiO3 perovskite",
    "neph": "Nepheline",
    "ky": "Kyanite",
    "q": "Quartz",
    "coe": "Coesite",
    "st": "Stishovite",
    "seif": "SiO2 with the alpha-PbO2 structure",
    "wo": "Wollastonite",
    "pwo": "Pseudowollastonite",
    "Fe-bcc": "Alpha (bcc) metallic iron",
    "Fe-fcc": "Gamma (fcc) metallic iron",
    "Fe-hcp": "Epsilon (hcp) metallic iron",
}


def candidate_phases():
    solutions = [
        (SLB24.olivine, "ol"),
        (SLB24.wadsleyite, "wa"),
        (SLB24.ringwoodite, "ri"),
        (SLB24.orthopyroxene, "opx"),
        (SLB24.clinopyroxene, "cpx"),
        (SLB24.c2c_pyroxene, "hpcpx"),
        (SLB24.garnet, "gt"),
        (SLB24.mg_fe_aluminous_spinel, "sp"),
        (SLB24.plagioclase, "pl"),
        (SLB24.ilmenite, "ak"),
        (SLB24.ferropericlase, "fp"),
        (SLB24.bridgmanite, "bg"),
        (SLB24.post_perovskite, "ppv"),
        (SLB24.calcium_ferrite_structured_phase, "cf"),
        (SLB24.new_aluminous_phase, "nal"),
    ]
    pure = [
        (SLB24.capv, "capv"),
        (SLB24.neph, "neph"),
        (SLB24.ky, "ky"),
        (SLB24.qtz, "q"),
        (SLB24.coes, "coe"),
        (SLB24.st, "st"),
        (SLB24.apbo, "seif"),
        (SLB24.wo, "wo"),
        (SLB24.pwo, "pwo"),
        (SLB24.fea, "Fe-bcc"),
        (SLB24.feg, "Fe-fcc"),
        (SLB24.fee, "Fe-hcp"),
    ]
    phases = []
    for factory, name in solutions + pure:
        phase = factory()
        phase.set_name(name)
        phases.append(phase)
    return phases


def settings(seeds=9, quick=False, verbose=False):
    opts = bm.PseudosectionSettings()
    opts.pressure_seeds = opts.temperature_seeds = 4 if quick else seeds
    # Mg-Tschermak loses its stable volume at LP/HT while orthopyroxene
    # is still an essential competitor. Mask that conservative model limit.
    opts.required_eos_phases = ["opx"]
    opts.verbose = verbose
    return opts


def calculate(seeds=9, quick=False, verbose=False):
    return bm.pseudosection(
        PYROLITE_COMPOSITION.atomic_composition,
        candidate_phases(),
        (0.0, P_Pa),
        (0.0, T_K),
        settings(seeds, quick, verbose),
    )


def refine(source, verbose=False):
    data = json.loads(source.read_text())
    oxides = data.get("oxide_wt_percent")
    if not isinstance(oxides, dict) or oxides.keys() != PYROLITE_OXIDES.keys():
        raise ValueError("Saved diagram has a different bulk composition.")
    # Normalize through Composition and allow serialization/arithmetic roundoff
    # in older oxide metadata, while rejecting a materially different bulk.
    saved_composition = bm.Composition(oxides, unit_type="mass", normalize=True)
    saved_composition.renormalize("mass", "total", 0.1)
    saved_mass = saved_composition.mass_composition
    if any(
        not isclose(saved_mass[oxide], mass, rel_tol=1.0e-12, abs_tol=1.0e-15)
        for oxide, mass in PYROLITE_COMPOSITION.mass_composition.items()
    ):
        raise ValueError("Saved diagram has a different bulk composition.")
    opts = bm.PseudosectionResult.from_dict(data).settings
    opts.verbose = verbose
    return bm.refine_pseudosection(
        PYROLITE_COMPOSITION.atomic_composition, candidate_phases(), data, opts
    )


def save_json(result, path):
    data = result.to_dict()
    data.update(
        oxide_wt_percent=PYROLITE_OXIDES,
        closed_system=True,
        dataset="SLB24",
        dataset_source=PAPER_URL,
        initial_ferric_fraction=INITIAL_FERRIC_FRACTION,
        solid_only=True,
        display_pressure_range_GPa=[0.0, P_Pa / 1.0e9],
        display_temperature_range_K=[0.0, T_K],
    )
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def plot(result, path, label_fontsize=7.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(bottom=0.20)
    key_path = path.with_name(path.stem + "_assemblages.md")
    bm.plot_pseudosection(
        result,
        ax=ax,
        pressure_unit="GPa",
        temperature_unit="K",
        show_nodes=False,
        label_fontsize=label_fontsize,
        label_key_path=key_path,
    )
    if ax.get_legend() is not None:
        ax.get_legend().set_bbox_to_anchor((0.0, -0.11))
    ax.set(
        xlim=(0, T_K),
        ylim=(0, P_Pa / 1.0e9),
        xlabel="Temperature (K)",
        ylabel="Pressure (GPa)",
        title="Pyrolite, 3% Fe³⁺/ΣFe on a no metal-basis",
    )
    ax.grid(alpha=0.12)
    scope = "SLB2024 solids; closed oxygen; high/low spin populations equilibrated; no liquids"
    diagnostics = (
        result["diagnostics"] if isinstance(result, dict) else result.diagnostics
    )
    polygons = ax.pseudosection_geometry.polygons
    open_fields = sum(
        p.has_open_boundary and not p.outside_model_domain for p in polygons
    )
    unidentified = sum(
        (p.n_phases <= 0 or not p.phases) and not p.outside_model_domain
        for p in polygons
    )
    status = f"{open_fields} open fields; {unidentified} unidentified fields; {len(diagnostics)} numerical diagnostics in JSON"
    caption = scope + "\n" + status + "; finite seeds may miss narrow fields."
    caption += (
        "\nHatched area: outside the orthopyroxene EOS domain (Mg-Tschermak spinodal). Numbered fields: "
        + key_path.name
    )
    fig.text(0.1, 0.015, caption, fontsize=8)
    if key_path.exists():
        with key_path.open("a") as key:
            key.write("\n## Phase abbreviations\n\n| Label | Model |\n| --- | --- |\n")
            for label, name in PHASE_GLOSSARY.items():
                key.write(f"| {label} | {name} |\n")
            key.write(
                "\nA leading number means that many coexisting compositions of the same solution.\n"
            )
            key.write(
                f"\nDataset and bulk source: [Stixrude & Lithgow-Bertelloni (2024)]({PAPER_URL}).\n"
            )
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_saved_json(source, output_dir, label_fontsize=7.0):
    data = json.loads(source.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pyrolite_pseudosection.json"
    if source.resolve() != target.resolve():
        target.write_text(source.read_text())
    plot(data, output_dir / "pyrolite_pseudosection.png", label_fontsize)
    print(f"Coloured pseudosection figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=9)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=7.0,
        help="Font size in points for every assemblage and field number (default: 7).",
    )
    parser.add_argument(
        "--plot-json",
        type=Path,
        help="Replot a saved pyrolite result without repeating equilibrium solves.",
    )
    parser.add_argument(
        "--refine-json",
        type=Path,
        help="Resume unfinished lines from saved successful equilibrium states in C++.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pyrolite_pseudosection_output")
    )
    args = parser.parse_args()
    if args.plot_json:
        plot_saved_json(args.plot_json, args.output_dir, args.label_fontsize)
        return
    result = (
        refine(args.refine_json, args.verbose)
        if args.refine_json
        else calculate(args.seeds, args.quick, args.verbose)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, args.output_dir / "pyrolite_pseudosection.json")
    if not args.no_plots:
        plot(
            result, args.output_dir / "pyrolite_pseudosection.png", args.label_fontsize
        )
    print(
        f"{len(result.fields)} fields, {len(result.boundaries)} edges, "
        f'{sum(n.kind=="junction" for n in result.nodes)} junctions; '
        f"{result.equilibrium_solves} native equilibrium solves."
    )
    print(f"Resolved: {result.resolved}. Outputs: {args.output_dir}")
    for diagnostic in result.diagnostics:
        print(diagnostic)


if __name__ == "__main__":
    main()
