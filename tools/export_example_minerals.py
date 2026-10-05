#!/usr/bin/env python3
"""Developer utility: copy example published mineral data into C++.

This script requires the reference BurnMan package to regenerate the checked-in
catalogue. Neither the C++ build nor the examples execute this script or import
BurnMan. Compatibility factories for fixed reduced models are also exported;
the examples now use the general C++ polytope simplifier at runtime.
"""

import argparse
import difflib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import burnman
from burnman.minerals import (
    HP_2011_ds62 as HP,
    SLB_2011 as SLB,
    JH_2015 as JH,
    mb50NCKFMASHTO as MB,
    HGP_2018_ds633 as HGP,
)
from burnman.minerals import mp50NCKFMASHTO as MP
from burnman.minerals import SLB_2024 as SLB24
from burnman.classes import solutionmodel as models
from burnman.tools.solution import transform_solution_to_new_basis

from burnman_reference import REFERENCE, verify_reference


def literal(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                "{" + literal(k) + ", " + literal(v) + "}"
                for k, v in sorted(value.items())
            )
            + "}"
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return "{" + ", ".join(literal(v) for v in value) + "}"
    value = float(value)
    if math.isnan(value):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(value):
        return ("-" if value < 0 else "") + "std::numeric_limits<double>::infinity()"
    return repr(value)


class Exporter:
    def __init__(self):
        self.functions = []
        self.minerals = {}

    def mineral(self, phase):
        """Deduplicate minerals by complete thermodynamic definition."""
        if not isinstance(phase, burnman.CombinedMineral) and phase.params["n"] != int(
            phase.params["n"]
        ):
            # Staurolite is tabulated per 81.5 atoms. The native thermal API
            # takes an integer atom count. Evaluate HP per one-atom formula
            # unit, then restore the original molar unit with CombinedMineral.
            # All extensive properties and formula amounts scale together;
            # intensives (including K, alpha and Einstein T) remain identical.
            if (
                phase.params["equation_of_state"] != "hp_tmt"
                or phase.property_modifiers
            ):
                raise ValueError(
                    "Fractional formula-unit export requires an unmodified HP mineral."
                )
            count = phase.params["n"]
            params = dict(phase.params)
            for quantity in ("H_0", "S_0", "V_0", "F_0", "E_0", "molar_mass"):
                if quantity in params:
                    params[quantity] /= count
            params["Cp"] = (np.asarray(params["Cp"]) / count).tolist()
            params["formula"] = {
                element: amount / count for element, amount in params["formula"].items()
            }
            params["n"] = 1
            params["name"] = phase.name + " (one-atom formula unit)"
            return self.mineral(
                burnman.CombinedMineral(
                    [burnman.Mineral(params)], [count], [0.0, 0.0, 0.0], phase.name
                )
            )
        if isinstance(phase, burnman.CombinedMineral):
            components = [self.mineral(m) for m, _ in phase.mixture.endmembers]
            key = (
                phase.name,
                tuple(components),
                literal(phase.mixture.molar_fractions),
                literal(phase.property_modifiers),
            )
            setup = [
                f"  Mineral m = make_combined_mineral({{{', '.join(f'{c}()' for c in components)}}},",
                f"    array({literal(phase.mixture.molar_fractions)}), Eigen::Vector3d::Zero(), {literal(phase.name)});",
            ]
        else:
            params = {
                k: v
                for k, v in phase.params.items()
                if k not in ("param_uncertainties", "property_modifiers", "Z")
            }
            # The 2024 stishovite EOS adds the published shear softening.
            if isinstance(params["equation_of_state"], SLB24.SLB3Stishovite):
                params["equation_of_state"] = "slb3-stishovite"
            key = (phase.name, literal(params), literal(phase.property_modifiers))
            setup = ["  Mineral m;"]
            for name, value in sorted(params.items()):
                if name == "equation_of_state":
                    enums = {
                        "hp_tmt": "HPTMT",
                        "hp_tmtL": "HPTMTL",
                        "slb3": "SLB3",
                        "slb3-conductive": "SLB3Conductive",
                        "slb3-stishovite": "SLB3Stishovite",
                    }
                    setup.append(
                        f"  m.params.equation_of_state = types::EOSType::{enums[value]};"
                    )
                elif name == "n":
                    assert value == int(value)
                    setup.append(f"  m.params.napfu = {int(value)};")
                elif name == "Cp":
                    setup.append(f"  m.params.Cp = types::CpParams{literal(value)};")
                elif name == "formula":
                    setup.append(
                        f"  m.params.formula = types::FormulaMap{literal(value)};"
                    )
                else:
                    field = {"Debye_0": "debye_0"}.get(name, name)
                    setup.append(f"  m.params.{field} = {literal(value)};")
            setup.append("  m.set_method(types::EOSType::Auto);")
            setup.append(f"  m.set_name({literal(phase.name)});")
        if key in self.minerals:
            return self.minerals[key]
        name = f"endmember_{len(self.minerals)}"
        self.minerals[key] = name
        modifiers = []
        for kind, p in phase.property_modifiers:
            if kind == "linear":
                modifiers.append(
                    "eos::excesses::LinearParams"
                    + literal([p["delta_V"], p["delta_S"], p["delta_E"]])
                )
            elif kind == "landau_hp":
                modifiers.append(
                    "eos::excesses::LandauHPParams"
                    + literal([p[k] for k in ["T_0", "P_0", "Tc_0", "V_D", "S_D"]])
                )
            elif kind == "landau_slb_2022":
                modifiers.append(
                    "eos::excesses::LandauSLB2022Params"
                    + literal([p[k] for k in ["Tc_0", "V_D", "S_D"]])
                )
            elif kind == "magnetic_chs":
                modifiers.append(
                    "eos::excesses::MagneticChsParams"
                    + literal(
                        [
                            p["structural_parameter"],
                            *p["curie_temperature"],
                            *p["magnetic_moment"],
                        ]
                    )
                )
            elif kind == "bragg_williams":
                assert p["n"] == int(p["n"])
                modifiers.append(
                    "eos::excesses::BraggWilliamsParams{"
                    + str(int(p["n"]))
                    + ", "
                    + ", ".join(
                        literal(p[k])
                        for k in ["factor", "Wh", "Wv", "deltaH", "deltaV"]
                    )
                    + "}"
                )
            else:
                raise ValueError(f"Unsupported modifier: {kind}")
        if modifiers:
            setup.append(
                f"  m.set_property_modifier_params({{{', '.join(modifiers)}}});"
            )
        setup.append("  return m;")
        self.functions.append(f"Mineral {name}() {{\n" + "\n".join(setup) + "\n}\n")
        return name

    def solution(self, phase, name):
        model = phase.solution_model
        # The reference HGP melt spells the jadeite pseudo-species Alsi2.
        # The formula parser interprets its trailing 2 as an occupancy count,
        # whereas the original THERMOCALC definition above it specifies pjd=1.
        # A letter-only species name retains that unit occupancy and makes
        # the Gibbs energy, chemical potentials and Hessian consistent.
        members = [
            (self.mineral(m), sites.replace("[Alsi2]", "[Alsitwo]"))
            for m, sites in model.endmembers
        ]
        setup = [
            "  types::PairedEndmemberList members = {",
            ",\n".join(f"    {{{m}(), {literal(sites)}}}" for m, sites in members),
            "  };",
        ]
        if isinstance(model, models.AsymmetricRegularSolution):
            alphas = np.asarray(model.alphas)

            def interaction(matrix):
                raw = np.asarray(matrix) * (alphas[:, None] + alphas[None, :]) / 2.0
                return [raw[i, i + 1 :].tolist() for i in range(len(members) - 1)]

            kind = (
                "SymmetricRegularSolution"
                if isinstance(model, models.SymmetricRegularSolution)
                else "AsymmetricRegularSolution"
            )
            args = "members, " + (
                "std::vector<double>" + literal(alphas) + ", "
                if kind.startswith("Asymmetric")
                else ""
            )
            args += ",\n    ".join(
                "Interactions" + literal(interaction(matrix))
                for matrix in [model.We, model.Wv, model.Ws]
            )
            setup.append(
                f"  auto model = std::make_shared<solution_models::{kind}>(\n    {args});"
            )
        elif type(model) is models.IdealSolution:
            setup.append(
                "  auto model = std::make_shared<solution_models::IdealSolution>(members);"
            )
        else:
            raise ValueError(f"Unsupported model: {type(model)}")
        setup.extend(
            [
                "  auto solution = std::make_shared<Solution>();",
                "  solution->set_solution_model(model);",
                f"  solution->set_name({literal(phase.name)});",
                f"  solution->set_composition(Eigen::ArrayXd::Constant({len(members)}, 1.0 / {len(members)}.0));",
                "  return solution;",
            ]
        )
        return f"std::shared_ptr<Solution> {name}() {{\n" + "\n".join(setup) + "\n}\n"


def reduced(phase, basis):
    """Export a fixed compatibility model in its intended endmember basis.

    The general polytope simplifier selects independent rows by sorting QP
    weights. Equal weights can exchange order with solver/platform roundoff,
    or select another equivalent basis. These three legacy factories need
    reproducible definitions; their parameters still come exclusively from
    the pinned Python BurnMan basis transformation.
    """
    solution = transform_solution_to_new_basis(
        phase, basis, solution_name=f"{phase.name} (transformed)"
    )
    # Retain the names used by the reference polytope simplifier.
    for i, name in enumerate(solution.endmember_names):
        if name == "User-created endmember":
            name = f"Derived member (occupancies: {solution.endmembers[i][1]})"
            solution.endmembers[i][0].name = name
            solution.endmember_names[i] = name
    return solution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the catalogue with the pinned reference without changing files.",
    )
    args = parser.parse_args()
    verify_reference(burnman)
    exporter = Exporter()
    definitions = []
    declarations = []
    bindings = []
    groups = {
        "HP_2011_ds62": [
            ("sill", HP.sill()),
            ("andalusite", HP.andalusite()),
            ("ky", HP.ky()),
        ]
        + [
            (name, getattr(HP, name)())
            for name in [
                "q",
                "law",
                "zo",
                "ru",
                "sph",
                "ab",
                "ta",
                "pre",
                "pa",
                "ma",
                "coe",
            ]
        ],
        "HGP_2018_ds633": [
            (
                "silicate_melt",
                HGP.make_melt_class(
                    [
                        HGP.q4L,
                        HGP.sl1L,
                        HGP.wo1L,
                        HGP.fo2L,
                        HGP.fa2L,
                        HGP.jdL,
                        HGP.hmL,
                        HGP.tiL,
                        HGP.kjL,
                        HGP.ctL,
                        HGP.h2o1L,
                    ]
                )(),
            )
        ],
        "mb50NCKFMASHTO": [
            (name, getattr(MB, name)())
            for name in [
                "hb",
                "aug",
                "dio",
                "opx",
                "g",
                "ol",
                "pl4tr",
                "abc",
                "k4tr",
                "ksp",
                "plc",
                "pli",
                "sp",
                "ilm",
                "ilmm",
                "ep",
                "bi",
                "mu",
                "chl",
            ]
        ],
        "SLB_2011": [
            (name, getattr(SLB, name)())
            for name in [
                "mg_fe_olivine",
                "mg_fe_wadsleyite",
                "mg_fe_ringwoodite",
                "mg_fe_bridgmanite",
                "post_perovskite",
                "ferropericlase",
                "orthopyroxene",
                "garnet",
                "ca_perovskite",
            ]
        ]
        + [
            (
                "pyrope_grossular",
                reduced(SLB.garnet(), [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
            ),
            (
                "mg_fe_bridgmanite_binary",
                reduced(SLB.mg_fe_bridgmanite(), [[1, 0, 0], [0, 1, 0]]),
            ),
        ],
        "JH_2015": [
            ("orthopyroxene", JH.orthopyroxene()),
            (
                "mg_fe_orthopyroxene",
                # en, en + fs - ordered ferroenstatite, fs.
                reduced(
                    JH.orthopyroxene(),
                    [
                        [1, 0, 0, 0, 0, 0, 0],
                        [1, 1, -1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                    ],
                ),
            ),
        ],
        "mp50NCKFMASHTO": [
            (name, getattr(MP, name)())
            for name in [
                "g",
                "pl4tr",
                "k4tr",
                "plc",
                "ksp",
                "ep",
                "ma",
                "mu",
                "bi",
                "opx",
                "sa",
                "cd",
                "st",
                "chl",
                "ctd",
                "sp",
                "ilmm",
                "ilm",
                "mt1",
            ]
        ],
        "SLB_2024": [
            (name, getattr(SLB24, name)())
            for name in [
                "c2c_pyroxene",
                "calcium_ferrite_structured_phase",
                "clinopyroxene",
                "garnet",
                "ilmenite",
                "ferropericlase",
                "new_aluminous_phase",
                "olivine",
                "orthopyroxene",
                "plagioclase",
                "post_perovskite",
                "bridgmanite",
                "ringwoodite",
                "mg_fe_aluminous_spinel",
                "wadsleyite",
            ]
        ]
        + [
            (name, getattr(SLB24, name)())
            for name in [
                "ab",
                "acm",
                "al",
                "alpv",
                "an",
                "anao",
                "andr",
                "apbo",
                "appv",
                "capv",
                "cats",
                "cen",
                "co",
                "coes",
                "cppv",
                "crcf",
                "crpv",
                "di",
                "en",
                "esk",
                "fa",
                "fapv",
                "fea",
                "fec2",
                "fecf",
                "fee",
                "feg",
                "feil",
                "fepv",
                "feri",
                "fewa",
                "fnal",
                "fo",
                "fppv",
                "fs",
                "gr",
                "hc",
                "he",
                "hem",
                "hepv",
                "hlpv",
                "hmag",
                "hppv",
                "jd",
                "knor",
                "ky",
                "lppv",
                "mag",
                "mgc2",
                "mgcf",
                "mgil",
                "mgmj",
                "mgpv",
                "mgri",
                "mgts",
                "mgwa",
                "mnal",
                "mppv",
                "nacf",
                "namj",
                "neph",
                "nnal",
                "odi",
                "pe",
                "picr",
                "pwo",
                "py",
                "qtz",
                "smag",
                "sp",
                "st",
                "wo",
                "wu",
                "wuls",
            ]
        ],
    }
    for group, phases in groups.items():
        functions = []
        headers = []
        bind = [f'  auto {group} = catalog.def_submodule("{group}");']
        for name, phase in phases:
            if isinstance(phase, burnman.Solution):
                functions.append(exporter.solution(phase, name))
                headers.append(f"std::shared_ptr<Solution> {name}();")
            else:
                mineral = exporter.mineral(phase)
                functions.append(
                    f"std::shared_ptr<Mineral> {name}() {{ return std::make_shared<Mineral>({mineral}()); }}\n"
                )
                headers.append(f"std::shared_ptr<Mineral> {name}();")
            bind.append(f'  {group}.def("{name}", &minerals::{group}::{name});')
        definitions.append(
            f"namespace {group} {{\n"
            + "\n".join(functions)
            + f"}} // namespace {group}\n"
        )
        declarations.append(
            f"namespace {group} {{\n"
            + "\n".join(headers)
            + f"\n}} // namespace {group}\n"
        )
        bindings.extend(bind)
    notice = (
        "/*\n * Published mineral data copied from BurnMan, copyright (C) 2012–2025\n * by the BurnMan team, GPL v2 or later. This adaptation is GPL v3 or later.\n * Generated by tools/export_example_minerals.py; no Python is used at runtime.\n * Python BurnMan commit: "
        + REFERENCE["commit"]
        + ".\n * Sources: HP_2011_ds62 (Holland & Powell 2011, dataset 6.2),\n * SLB_2011 (Stixrude & Lithgow-Bertelloni 2011), and\n * JH_2015 (Jennings & Holland 2015, doi:10.1093/petrology/egv020),\n * mb50NCKFMASHTO (Green et al. 2016), mp50NCKFMASHTO metapelite models,\n * HGP_2018_ds633 (Holland et al. 2018), and SLB_2024 (Stixrude &\n * Lithgow-Bertelloni 2024, doi:10.1093/gji/ggae126).\n * Melt pseudo-species Alsi2 is renamed Alsitwo to preserve pjd=1.\n */\n"
    )
    cpp = (
        notice
        + '#include "burnman/minerals/datasets.hpp"\n#include "burnman/core/combined_mineral.hpp"\n#include "burnman/core/solution_model.hpp"\n#include <limits>\n\nnamespace burnman::minerals {\nnamespace {\nusing Interactions = std::vector<std::vector<double>>;\nEigen::ArrayXd array(std::initializer_list<double> values) {\n  return Eigen::Map<const Eigen::ArrayXd>(values.begin(), static_cast<Eigen::Index>(values.size()));\n}\n\n'
    )
    cpp += (
        "\n".join(exporter.functions)
        + "} // namespace\n\n"
        + "\n".join(definitions)
        + "} // namespace burnman::minerals\n"
    )
    header = (
        notice
        + '#pragma once\n#include "burnman/core/mineral.hpp"\n#include "burnman/core/solution.hpp"\n\nnamespace burnman::minerals {\n'
        + "\n".join(declarations)
        + "} // namespace burnman::minerals\n"
    )
    binding = (
        '#include "bindings.hpp"\n#include "burnman/minerals/datasets.hpp"\n\nnamespace burnman::python {\nvoid bind_minerals(py::module_& m) {\n  auto catalog = m.def_submodule("minerals", "Native mineral factories for the equilibration examples.");\n'
        + "\n".join(bindings)
        + "\n}\n} // namespace burnman::python\n"
    )
    mismatches = []
    for filename, text in [
        ("src/minerals/datasets.cpp", cpp),
        ("include/burnman/minerals/datasets.hpp", header),
        ("python/bindings/minerals.cpp", binding),
    ]:
        path = args.output_root / filename
        if args.check:
            command = ["clang-format", "--style=file", f"--assume-filename={path}"]
            expected = subprocess.check_output(command, input=text, text=True)
            actual = subprocess.check_output(command, input=path.read_text(), text=True)
            if actual != expected:
                mismatches.append(filename)
                diff = list(
                    difflib.unified_diff(
                        actual.splitlines(keepends=True),
                        expected.splitlines(keepends=True),
                        fromfile=filename,
                        tofile=f"{filename} (pinned Python BurnMan)",
                    )
                )
                sys.stderr.write("".join(diff[:80]))
                if len(diff) > 80:
                    print(f"... {len(diff) - 80} further diff lines", file=sys.stderr)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if mismatches:
        raise SystemExit(
            "Catalogue differs from pinned Python BurnMan: " + ", ".join(mismatches)
        )
    print(
        f"{'Verified' if args.check else 'Exported'} {len(exporter.minerals)} endmembers and {sum(map(len, groups.values()))} factories from BurnMan {burnman.__version__}."
    )


if __name__ == "__main__":
    main()
