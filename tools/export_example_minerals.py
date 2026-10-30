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

"""Developer utility: copy complete published mineral datasets into C++.

This script requires the reference BurnMan package to regenerate the checked-in
catalogue. Neither the C++ build nor the examples execute this script or import
BurnMan. Compatibility factories for fixed reduced models are also exported;
the examples now use the general C++ polytope simplifier at runtime.
"""

import argparse
import difflib
import json
import inspect
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import burnman
from burnman.minerals import (
    HP_2011_ds62,
    SLB_2011,
    JH_2015,
    mb50NCKFMASHTO,
    HGP_2018_ds633,
)
from burnman.minerals import mp50NCKFMASHTO
from burnman.minerals import mp50MnNCKFMASHTO
from burnman.minerals import SLB_2024
from burnman.minerals import ig50NCKFMASHTOCr
from burnman.minerals import HPx_ds636, ig51W24, ig51G25
from burnman.minerals import DKS_2013_solids, DKS_2013_liquids
from burnman.minerals import SLB_2005, SLB_2011_ZSB_2013, SLB_2022
from burnman.minerals import HHPH_2013, HP_2011_fluids, ig50NCKFMASTOCr, mp50KFMASH
from burnman.minerals import (
    ICL_2018,
    KMFBZ_2017,
    Matas_etal_2007,
    Murakami_2013,
    Murakami_etal_2012,
    RS_2014_liquids,
    Sundman_1991,
    other,
)
from burnman.classes import solutionmodel as models
from burnman.tools.solution import transform_solution_to_new_basis

from burnman_reference import ROOT, verify_reference

CPP_HEADER = (ROOT / "contrib/utilities/cpp_header.txt").read_text() + "\n"
COVARIANCE_DATASETS = {"HP_2011_ds62", "HGP_2018_ds633", "HPx_ds636"}


def literal(value):
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
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


def is_hp2011_water(params):
    if params["equation_of_state"] != "pitzer-sterner":
        return False
    expected = burnman.minerals.HP_2011_fluids.H2O().params
    return params.keys() == expected.keys() and all(
        key == "name"
        or (
            value == expected[key]
            if isinstance(value, (str, dict))
            else np.allclose(value, expected[key], rtol=1.0e-12, atol=0.0)
        )
        for key, value in params.items()
    )


class Exporter:
    def __init__(self):
        self.functions = []
        self.minerals = {}

    def mineral(self, phase):
        """Deduplicate minerals by complete thermodynamic definition."""
        modifier_params = [(kind, dict(p)) for kind, p in phase.property_modifiers]
        if type(phase).__module__ == SLB_2022.__name__:
            # Convert HeFESTo critical volumes from cm^3/mol to m^3/mol.
            for kind, p in modifier_params:
                if kind == "landau_slb_2022" and p["V_D"] in (0.8, 1.35936):
                    p["V_D"] /= 1.0e6
        if isinstance(phase, burnman.CombinedMineral):
            components = [self.mineral(m) for m, _ in phase.mixture.endmembers]
            key = (
                phase.name,
                tuple(components),
                literal(phase.mixture.molar_fractions),
                literal(modifier_params),
            )
            setup = [
                f"  Mineral m = make_combined_mineral({{{', '.join(f'{c}()' for c in components)}}},",
                f"    array({literal(phase.mixture.molar_fractions)}), Eigen::Vector3d::Zero(), {literal(phase.name)});",
            ]
        elif is_hp2011_water(phase.params):
            # Dataset 6.36 uses the same HP2011 thermal reference as native water.
            key = (phase.name, literal(phase.params))
            setup = [
                "  Mineral m = *water_fluid();",
                f"  m.set_name({literal(phase.name)});",
            ]
        else:
            params = {
                k: v
                for k, v in phase.params.items()
                if k not in ("param_uncertainties", "property_modifiers")
            }
            # The 2024 stishovite EOS adds the published shear softening.
            if isinstance(params["equation_of_state"], SLB_2024.SLB3Stishovite):
                params["equation_of_state"] = "slb3-stishovite"
            key = (phase.name, literal(params), literal(modifier_params))
            setup = ["  Mineral m;"]
            for name, value in sorted(params.items()):
                if name == "equation_of_state":
                    enums = {
                        "vinet": "Vinet",
                        "bm3": "BM3",
                        "bm4": "BM4",
                        "mgd2": "MGD2",
                        "mgd3": "MGD3",
                        "hp_tmt": "HPTMT",
                        "hp_tmtL": "HPTMTL",
                        "slb2": "SLB2",
                        "slb3": "SLB3",
                        "slb3-conductive": "SLB3Conductive",
                        "slb3-stishovite": "SLB3Stishovite",
                        "dks_s": "DKS_S",
                        "dks_l": "DKS_L",
                        "cork": "CORK",
                        "pitzer-sterner": "PitzerSterner",
                    }
                    setup.append(
                        f"  m.params.equation_of_state = types::EOSType::{enums[value]};"
                    )
                elif name == "Cp":
                    setup.append(f"  m.params.Cp = types::CpParams{literal(value)};")
                elif name == "cork_params":
                    coefficients = [*value[0], value[1][0], *value[2], *value[3]]
                    setup.append(
                        f"  m.params.cork_params = types::CorkParams{literal(coefficients)};"
                    )
                elif name == "c_coeffs":
                    setup.append(
                        f"  m.params.c_coeffs = types::PitzerSternerCoefficients{{{literal(value)}}};"
                    )
                elif name == "dcorkdT":
                    setup.append(
                        f"  m.params.dcorkdT = std::array<double, 4>{literal(value)};"
                    )
                elif name == "formula":
                    setup.append(
                        f"  m.params.formula = types::FormulaMap{literal(value)};"
                    )
                elif name == "a":
                    setup.append(
                        f"  m.params.a = std::vector<std::vector<double>>{literal(value)};"
                    )
                else:
                    field = {
                        "Debye_0": "debye_0",
                        "n": "napfu",
                        "Kprime_prime_0": "Kdprime_0",
                        "O_theta": "order_theta",
                        "O_f": "order_f",
                    }.get(name, name)
                    if field in ("order_theta", "order_f"):
                        setup.append(f"  m.params.{field} = {int(value)};")
                        continue
                    setup.append(f"  m.params.{field} = {literal(value)};")
            setup.append("  m.set_method(types::EOSType::Auto);")
            setup.append(f"  m.set_name({literal(phase.name)});")
        uncertainties = getattr(phase, "uncertainties", None)
        if uncertainties is not None:
            key += (literal(uncertainties),)
            setup.append(
                f"  m.uncertainties = std::unordered_map<std::string, double>{literal(uncertainties)};"
            )
        if key in self.minerals:
            return self.minerals[key]
        name = f"endmember_{len(self.minerals)}"
        self.minerals[key] = name
        modifiers = []
        for kind, p in modifier_params:
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
            elif kind == "landau":
                modifiers.append(
                    "eos::excesses::LandauParams"
                    + literal([p[k] for k in ["Tc_0", "V_D", "S_D"]])
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
        if isinstance(phase, burnman.RelaxedSolution):
            parent = type(phase.unrelaxed).__name__
            return (
                f"std::shared_ptr<Solution> {name}() {{\n"
                f"  return std::make_shared<RelaxedSolution>(*{parent}(),\n"
                f"    matrix({literal(phase.dndq.T)}), matrix({literal(phase.dndx.T)}));\n}}\n"
            )
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


def dataset_entries(module):
    """Public mineral/solution classes, aliases and preconstructed endmembers."""
    entries = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if (
            inspect.isclass(value)
            and issubclass(value, burnman.Mineral)
            and value.__module__ == module.__name__
        ):
            phase = value()
            entries.append((name, phase))
        elif isinstance(value, burnman.Mineral):
            entries.append((name, value))
    return entries


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
        name: dataset_entries(module)
        for name, module in (
            ("HP_2011_ds62", HP_2011_ds62),
            ("HGP_2018_ds633", HGP_2018_ds633),
            ("mb50NCKFMASHTO", mb50NCKFMASHTO),
            ("SLB_2011", SLB_2011),
            ("JH_2015", JH_2015),
            ("mp50NCKFMASHTO", mp50NCKFMASHTO),
            ("SLB_2024", SLB_2024),
            ("ig50NCKFMASHTOCr", ig50NCKFMASHTOCr),
            ("HPx_ds636", HPx_ds636),
            ("ig51W24", ig51W24),
            ("ig51G25", ig51G25),
            ("DKS_2013_solids", DKS_2013_solids),
            ("DKS_2013_liquids", DKS_2013_liquids),
            ("SLB_2005", SLB_2005),
            ("SLB_2011_ZSB_2013", SLB_2011_ZSB_2013),
            ("SLB_2022", SLB_2022),
            ("HHPH_2013", HHPH_2013),
            ("HP_2011_fluids", HP_2011_fluids),
            ("ig50NCKFMASTOCr", ig50NCKFMASTOCr),
            ("mp50KFMASH", mp50KFMASH),
            ("mp50MnNCKFMASHTO", mp50MnNCKFMASHTO),
            ("ICL_2018", ICL_2018),
            ("KMFBZ_2017", KMFBZ_2017),
            ("Matas_etal_2007", Matas_etal_2007),
            ("Murakami_2013", Murakami_2013),
            ("Murakami_etal_2012", Murakami_etal_2012),
            ("RS_2014_liquids", RS_2014_liquids),
            ("Sundman_1991", Sundman_1991),
            ("other", other),
        )
    }
    # AA liquid iron has its own native factory in other.cpp.
    groups["other"] = [
        (name, phase) for name, phase in groups["other"] if name != "liquid_iron"
    ]
    # Retain the fixed reduced factories used before the complete catalogues.
    groups["SLB_2011"] += [
        (
            "pyrope_grossular",
            reduced(SLB_2011.garnet(), [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
        ),
        (
            "mg_fe_bridgmanite_binary",
            reduced(SLB_2011.mg_fe_bridgmanite(), [[1, 0, 0], [0, 1, 0]]),
        ),
    ]
    groups["JH_2015"].append(
        (
            "mg_fe_orthopyroxene",
            reduced(
                JH_2015.orthopyroxene(),
                [[1, 0, 0, 0, 0, 0, 0], [1, 1, -1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
            ),
        )
    )
    for group, phases in groups.items():
        functions = []
        headers = []
        bind = [f'  auto {group} = catalog.def_submodule("{group}");']
        if group == "SLB_2024":
            bind.append(
                '  SLB_2024.attr("phase_aliases") = minerals::SLB_2024::phase_aliases();'
            )
        aliases = {}
        for name, phase in phases:
            if isinstance(phase, burnman.Solution):
                if type(phase) is not burnman.Solution and type(phase) in aliases:
                    functions.append(
                        f"std::shared_ptr<Solution> {name}() {{ return {aliases[type(phase)]}(); }}\n"
                    )
                else:
                    functions.append(exporter.solution(phase, name))
                    aliases[type(phase)] = name
                headers.append(f"std::shared_ptr<Solution> {name}();")
            else:
                mineral = exporter.mineral(phase)
                functions.append(
                    f"std::shared_ptr<Mineral> {name}() {{ return std::make_shared<Mineral>({mineral}()); }}\n"
                )
                headers.append(f"std::shared_ptr<Mineral> {name}();")
            bind.append(f'  {group}.def("{name}", &minerals::{group}::{name});')
        if group in COVARIANCE_DATASETS:
            headers.append("CovarianceData cov();")
            bind.append(
                f'  {group}.def("cov", []() {{ return covariance_dict(minerals::{group}::cov()); }},'
                ' "Return endmember zero-point energy covariances in (J/mol)². '
                "The endmember_names list gives the row and column ordering. "
                'Each call returns independent data.");'
            )
        if group == "other":
            bind.append(
                '  other.def("liquid_iron", &minerals::other::liquid_iron, '
                '"AA1994 liquid Fe with the Figure 5 heat capacity and unfitted S_0=100 J/(mol K).");'
            )
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
    notice = CPP_HEADER + (
        "/*\n * Published mineral data copied from Python BurnMan.\n * Generated by tools/export_example_minerals.py; no Python is used at runtime.\n * Reference policy: tests/reference/burnman.json.\n * Sources: HP_2011_ds62 (Holland & Powell 2011, dataset 6.2),\n * SLB_2011 (Stixrude & Lithgow-Bertelloni 2011), and\n * JH_2015 (Jennings & Holland 2015, doi:10.1093/petrology/egv020),\n * mb50NCKFMASHTO (Green et al. 2016), mp50NCKFMASHTO metapelite models,\n * mp50MnNCKFMASHTO Mn-bearing metapelite models,\n * HGP_2018_ds633 and ig50NCKFMASHTOCr (Holland et al. 2018),\n * HPx_ds636, ig51W24 (Weller et al. 2024) and ig51G25 (Green et al. 2025),\n * SLB_2024 (Stixrude & Lithgow-Bertelloni 2024, doi:10.1093/gji/ggae126),\n * DKS_2013_solids and DKS_2013_liquids (de Koker et al. 2013, doi:10.1016/j.epsl.2012.11.026).\n * SLB_2005, SLB_2011_ZSB_2013 and SLB_2022 (Stixrude & Lithgow-Bertelloni,\n * and Zhang, Stixrude & Brodholt 2013).\n * HHPH_2013 (Holland et al. 2013), HP_2011_fluids (Holland & Powell 2011),\n * ig50NCKFMASTOCr (Holland et al. 2018) and mp50KFMASH metapelite models.\n * ICL_2018 (Irving, Cottaar & Lekic 2018), KMFBZ_2017 (Kurnosov et al. 2017),\n * Matas_etal_2007, Murakami_2013 and Murakami_etal_2012,\n * RS_2014_liquids (Ramo & Stixrude 2014), Sundman_1991 and other mineral models.\n * Legacy HGP_2018_ds633 melt pseudo-species Alsi2 is renamed Alsitwo to preserve pjd=1.\n */\n"
    )
    cpp = (
        notice
        + '#include "burnman/minerals/datasets.hpp"\n#include "burnman/minerals/water.hpp"\n#include "burnman/core/combined_mineral.hpp"\n#include "burnman/core/solution_model.hpp"\n#include "burnman/core/relaxed_solution.hpp"\n#include <limits>\n\nnamespace burnman::minerals {\nnamespace {\nusing Interactions = std::vector<std::vector<double>>;\nEigen::ArrayXd array(std::initializer_list<double> values) {\n  return Eigen::Map<const Eigen::ArrayXd>(values.begin(), static_cast<Eigen::Index>(values.size()));\n}\nEigen::MatrixXd matrix(std::initializer_list<std::initializer_list<double>> rows) {\n  Eigen::MatrixXd m(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(rows.begin()->size()));\n  Eigen::Index i = 0;\n  for (const auto &row : rows) m.row(i++) = array(row).matrix().transpose();\n  return m;\n}\n\n'
    )
    cpp += (
        "\n".join(exporter.functions)
        + "} // namespace\n\n"
        + "\n".join(definitions)
        + "} // namespace burnman::minerals\n"
    )
    header = (
        notice
        + '#pragma once\n#include "burnman/core/mineral.hpp"\n#include "burnman/core/solution.hpp"\n#include "burnman/minerals/covariance.hpp"\n#include "burnman/minerals/other.hpp"\n\nnamespace burnman::minerals {\n'
        + "\n".join(declarations)
        + "} // namespace burnman::minerals\n"
    )
    binding = CPP_HEADER + (
        '#include "bindings.hpp"\n#include "burnman/minerals/datasets.hpp"\n#include "burnman/minerals/phase_aliases.hpp"\n\nnamespace burnman::python {\nnamespace {\npy::dict covariance_dict(minerals::CovarianceData data) {\n  py::dict result;\n  result["endmember_names"] = std::move(data.endmember_names);\n  result["covariance_matrix"] = std::move(data.covariance_matrix);\n  return result;\n}\n} // namespace\nvoid bind_minerals(py::module_& m) {\n  auto catalog = m.def_submodule("minerals", "Native factories for complete published mineral datasets.");\n'
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
