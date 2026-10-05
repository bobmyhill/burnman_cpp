#!/usr/bin/env python3
"""Check or regenerate native numerical snapshots using pinned Python BurnMan.

Run with the clean reference checkout first on PYTHONPATH. --write replaces
the numerical literals in the existing C++ tests. Inputs come from those tests;
expected outputs come exclusively from Python BurnMan, never burnman_cpp.
Analytic identities and algorithm regression fixtures are outside this audit.
"""

import argparse
import ast
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import scipy.optimize
import burnman
from burnman.classes import averaging_schemes, solutionmodel
from burnman.eos import bukowinski_electronic, debye, einstein, property_modifiers
from burnman.utils.math import complete_basis

from burnman_reference import REFERENCE, ROOT, verify_reference

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TOKENS = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[{}]', re.S)
RENAMES = {"napfu": "n", "debye_0": "Debye_0"}


def numbers(text):
    return [float(value) for value in text.split(",") if value.strip()]


def block_end(text, start):
    depth = 0
    for token in TOKENS.finditer(text, start):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            if depth == 0:
                return token.end()
    raise ValueError("Unbalanced C++ test block")


def blocks(text, kind):
    pattern = (
        r'TEST_CASE(?:_METHOD)?\([^{}]*?"([^"\n]+)"[^{}]*?\)\s*\{'
        if kind == "test"
        else r'SECTION\("([^"\n]+)"\)\s*\{'
    )
    return [
        (match[1], match.start(), block_end(text, match.end() - 1))
        for match in re.finditer(pattern, text, re.S)
    ]


def parameters(text, variable="params"):
    params = {}
    for key, value in re.findall(
        rf"{re.escape(variable)}\.(\w+)\s*=\s*(.*?);", text, re.S
    ):
        if key == "equation_of_state":
            continue
        if key == "formula":
            value = {
                name: float(amount)
                for name, amount in re.findall(rf'\{{"(\w+)",\s*({NUMBER})\}}', value)
            }
        elif key == "Cp":
            value = numbers(value[value.index("{") + 1 : value.rindex("}")])
        else:
            value = ast.literal_eval(value.strip())
        params[RENAMES.get(key, key)] = value
    # The isothermal native EOS calls its reference energy E_0; Python uses F_0.
    if "E_0" in params:
        params.setdefault("F_0", params.pop("E_0"))
    return params


def make_eos(method, params):
    method = {"bm2": "bm3shear2"}.get(method, method)
    return burnman.Mineral(dict(params, equation_of_state=method))


def eos_value(mineral, key, pressure, temperature, volume):
    """Use upstream EOS functions at the test's supplied volume, including off-EOS states."""
    eos, params = mineral.method, mineral.params
    args = (pressure, temperature, volume, params)
    if key == "P":
        return eos.pressure(temperature, volume, params)
    if key == "V":
        return eos.volume(pressure, temperature, params)
    functions = {
        "KT": "isothermal_bulk_modulus_reuss",
        "K": "isothermal_bulk_modulus_reuss",
        "G2": "shear_modulus",
        "G3": "shear_modulus",
        "S": "entropy",
        "Cp": "molar_heat_capacity_p",
        "alpha": "thermal_expansivity",
        "G": "gibbs_energy",
    }
    if key in functions:
        return getattr(eos, functions[key])(*args)
    if key in ("F", "E", "H"):
        if hasattr(eos, "_helmholtz_energy"):
            helmholtz = eos._helmholtz_energy(*args)
        elif hasattr(eos, "_molar_helmholtz_energy"):
            helmholtz = eos._molar_helmholtz_energy(*args)
        else:
            helmholtz = eos.gibbs_energy(*args) - pressure * volume
        if key == "F":
            return helmholtz
        entropy = eos.entropy(*args)
        return (
            helmholtz + temperature * entropy + (pressure * volume if key == "H" else 0)
        )
    if key == "Cv":
        if hasattr(eos, "_molar_heat_capacity_v"):
            return eos._molar_heat_capacity_v(*args)
        cp = eos.molar_heat_capacity_p(*args)
        alpha = eos.thermal_expansivity(*args)
        kt = eos.isothermal_bulk_modulus_reuss(*args)
        return cp - alpha**2 * kt * volume * temperature
    if key == "KS":
        kt = eos.isothermal_bulk_modulus_reuss(*args)
        return (
            kt
            * eos.molar_heat_capacity_p(*args)
            / eos_value(mineral, "Cv", pressure, temperature, volume)
        )
    if key == "gamma":
        if hasattr(eos, "_grueneisen_parameter"):
            function = eos._grueneisen_parameter
            if len(inspect.signature(function).parameters) == 2:
                return function(params["V_0"] / volume, params)
            return function(*args)
        return (
            eos.thermal_expansivity(*args)
            * eos.isothermal_bulk_modulus_reuss(*args)
            * volume
            / eos_value(mineral, "Cv", pressure, temperature, volume)
        )
    raise ValueError(f"Unknown EOS reference property: {key}")


@contextmanager
def tight_volumes():
    original = scipy.optimize.brentq

    def tight(*args, **kwargs):
        kwargs["xtol"] = REFERENCE["volume_root_xtol_m3_per_mol"]
        return original(*args, **kwargs)

    scipy.optimize.brentq = tight
    try:
        yield
    finally:
        scipy.optimize.brentq = original


class Audit:
    def __init__(self):
        self.edits = {}
        self.records = []

    def expect(self, path, start, end, value, label, as_integer=False):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite reference: {path}: {label}")
        source = path.read_text()
        previous = float(source[start:end])
        literal = str(int(value)) if as_integer else repr(value)
        self.edits.setdefault(path, []).append((start, end, literal))
        self.records.append(
            dict(
                file=str(path.relative_to(ROOT)),
                label=label,
                value=value,
                previous=previous,
                agrees=math.isclose(previous, value, rel_tol=1e-12, abs_tol=1e-16),
            )
        )

    def scalars(self, path, offset, text, values):
        matched = set()
        for match in re.finditer(
            rf"\b(?:double|Eigen::Index)\s+(\w+)\s*=\s*({NUMBER})\s*;", text
        ):
            if match[1] in values:
                self.expect(
                    path,
                    offset + match.start(2),
                    offset + match.end(2),
                    values[match[1]],
                    match[1],
                    as_integer=match[0].startswith("Eigen::Index"),
                )
                matched.add(match[1])
        missing = set(values) - matched
        if missing:
            raise ValueError(f"Missing scalar targets in {path}: {missing}")

    def arrays(self, path, offset, text, values):
        matched = set()
        for match in re.finditer(r"\b(\w+)\s*<<\s*([^;]+);", text):
            if match[1] not in values:
                continue
            expected = np.asarray(values[match[1]], dtype=float).ravel()
            literals = list(re.finditer(NUMBER, match[2]))
            if len(literals) != len(expected):
                raise ValueError(f"Wrong array size: {path}: {match[1]}")
            for index, (literal, value) in enumerate(zip(literals, expected)):
                self.expect(
                    path,
                    offset + match.start(2) + literal.start(),
                    offset + match.start(2) + literal.end(),
                    value,
                    f"{match[1]}[{index}]",
                )
            matched.add(match[1])
        if set(values) != matched:
            raise ValueError(
                f"Missing array targets in {path}: {set(values) - matched}"
            )

    def table(self, path, offset, text, evaluate):
        for match in re.finditer(r"TestData\{([^{}]+)\}", text):
            literals = list(re.finditer(NUMBER, match[1]))
            inputs = float(literals[0][0])
            expected = evaluate(inputs)
            if len(expected) != len(literals) - 1:
                raise ValueError(f"Wrong table width in {path}")
            for index, (literal, value) in enumerate(zip(literals[1:], expected)):
                self.expect(
                    path,
                    offset + match.start(1) + literal.start(),
                    offset + match.start(1) + literal.end(),
                    value,
                    f"TestData({inputs})[{index}]",
                )

    def maps(self, path, offset, text, evaluate):
        inputs = {"P1": 1e9, "P2": 54e9, "T1": 800, "T2": 2500, "x1": 0.99, "x2": 0.4}
        # Resolve local declarations such as double T1=300, T2=800, T3=2500.
        for declaration in re.finditer(r"\bdouble\s+([^;]+);", text):
            for name, value in re.findall(
                rf"(\w+)\s*=\s*({NUMBER})(?=\s*[,;]|\s*$)", declaration[1]
            ):
                inputs[name] = float(value)
        for table in re.finditer(r"\b(ref_\w+)\s*=\s*\{(.*?)\};", text, re.S):
            for entry in re.finditer(
                rf"\{{\{{([^{{}}]+)\}},\s*({NUMBER})\}}", table[2]
            ):
                key = [inputs[name.strip()] for name in entry[1].split(",")]
                value = evaluate(table[1].removeprefix("ref_"), key)
                start = offset + table.start(2) + entry.start(2)
                self.expect(
                    path,
                    start,
                    offset + table.start(2) + entry.end(2),
                    value,
                    f"{table[1]}({entry[1]})",
                )

    def finish(self, write):
        failed = [record for record in self.records if not record["agrees"]]
        for record in failed[:30]:
            print(
                f"Different: {record['file']} {record['label']}: {record['previous']!r} -> {record['value']!r}"
            )
        if write:
            for path, edits in self.edits.items():
                text = path.read_text()
                for start, end, value in sorted(edits, reverse=True):
                    text = text[:start] + value + text[end:]
                text = re.sub(
                    r"// Reference values from Py burnman v2\.1\.1a0\n", "", text
                )
                text = re.sub(
                    r"  +// Upstream BurnMan main 0cf782d7, with volume xtol=1e-24 m\^3/mol\.\n",
                    "",
                    text,
                )
                notice = f"// Numerical snapshots: Python BurnMan {REFERENCE['commit']}.\n// Reproduce with tools/check_reference_data.py; see docs/reference_data.md.\n"
                if notice not in text:
                    index = text.index("#include")
                    text = text[:index] + notice + text[index:]
                path.write_text(text)
        counts = {
            str(path.relative_to(ROOT)): len(edits)
            for path, edits in self.edits.items()
        }
        print(
            json.dumps(
                dict(
                    commit=REFERENCE["commit"],
                    values=len(self.records),
                    files=counts,
                    outside_tolerance=len(failed),
                    written=write,
                ),
                indent=2,
            )
        )
        return not failed or write


def audit_components(audit):
    functions = {
        "thermal_energy": "thermal_energy",
        "molar_heat_capacity_v": "molar_heat_capacity_v",
        "helmholtz_free_energy": "helmholtz_energy",
        "entropy": "entropy",
        "dmolar_heat_capacity_v_dT": "dmolar_heat_capacity_v_dT",
    }
    for name, module in [("debye", debye), ("einstein", einstein)]:
        path = ROOT / f"tests/eos/test_{name}.cpp"
        text = path.read_text()
        case = next(
            case
            for case in blocks(text, "test")
            if "functions python reference" in case[0]
        )
        for section, start, end in blocks(text[case[1] : case[2]], "section"):
            function = getattr(module, functions[section.removeprefix("compute_")])
            audit.scalars(
                path,
                case[1] + start,
                text[case[1] + start : case[1] + end],
                {
                    "ref_a": function(2000.0, 543.0, 2),
                    "ref_b": function(300.0, 773.0, 1),
                },
            )
        if name == "debye":
            case = next(
                case for case in blocks(text, "test") if case[0] == "Debye function"
            )
            audit.scalars(
                path,
                case[1],
                text[case[1] : case[2]],
                dict(
                    zip(
                        ["ref_a", "ref_b", "ref_c"],
                        [debye.debye_fn_cheb(x) for x in [0.1, 1.0, 10.0]],
                    )
                ),
            )
    path = ROOT / "tests/eos/test_bukowinski_electronic.cpp"
    text = path.read_text()
    case = next(case for case in blocks(text, "test") if "python reference" in case[0])
    values = dict(
        helmholtz_el=bukowinski_electronic.helmholtz(1200, 2e-6, 300, 7e-6, 0.004, 1.5),
        pressure_el=bukowinski_electronic.pressure(1200, 2e-6, 300, 7e-6, 0.004, 1.5),
        entropy_el=bukowinski_electronic.entropy(1200, 2e-6, 7e-6, 0.004, 1.5),
        KT_over_V=bukowinski_electronic.KToverV(1200, 2e-6, 300, 7e-6, 0.004, 1.5),
        CV_over_T=bukowinski_electronic.CVoverT(2e-6, 7e-6, 0.004, 1.5),
        alpha_KT=bukowinski_electronic.aKT(1200, 2e-6, 7e-6, 0.004, 1.5),
    )
    for section, start, end in blocks(text[case[1] : case[2]], "section"):
        audit.scalars(
            path,
            case[1] + start,
            text[case[1] + start : case[1] + end],
            {"ref": values[section.removeprefix("compute_")]},
        )


def audit_eos(audit):
    methods = dict(
        birch_murnaghan="bm3",
        vinet="vinet",
        modified_tait="mt",
        mie_grueneisen_debye="mgd3",
        slb="slb3",
        hp="hp_tmt",
    )
    for filename, method in methods.items():
        path = ROOT / f"tests/eos/test_{filename}.cpp"
        text = path.read_text()
        case = next(
            case for case in blocks(text, "test") if "python reference" in case[0]
        )
        offset = case[1]
        body = text[offset : case[2]]
        for name, start, end in blocks(body, "section"):
            section = body[start:end]
            variable = (
                "params_b"
                if name.endswith(" B")
                else "params_a" if name.endswith(" A") else "params"
            )
            p = parameters(body[:start], variable)
            mineral = make_eos(method, p) if p else None
            if name.startswith("Test volume dependent functions"):
                keys = (
                    ["P"]
                    if method == "mt"
                    else ["P", "K", "E"] + (["G2", "G3"] if method == "bm3" else [])
                )

                def evaluate(ratio):
                    return [
                        eos_value(
                            make_eos("bm2", p) if key == "G2" else mineral,
                            key,
                            0.0,
                            300.0,
                            ratio * p["V_0"],
                        )
                        for key in keys
                    ]

                audit.table(path, offset + start, section, evaluate)
            elif name == "Test pressure dependent functions":
                audit.table(
                    path,
                    offset + start,
                    section,
                    lambda pressure: [
                        eos_value(mineral, key, pressure, 300.0, p["V_0"])
                        for key in ["KT", "G"]
                    ],
                )
            elif name == "V dependent functions":
                audit.table(
                    path,
                    offset + start,
                    section,
                    lambda ratio: [
                        eos_value(mineral, "gamma", 2e9, 500.0, ratio * p["V_0"])
                    ],
                )
            elif name == "Test Gibbs":
                values = {}
                for suffix, variable, ratio, pressure in [
                    ("aa", "params_a", 0.99, 1.6e9),
                    ("ab", "params_b", 0.99, 1.6e9),
                    ("ba", "params_a", 0.8, 54e9),
                    ("bb", "params_b", 0.8, 65e9),
                ]:
                    parameters_ = parameters(body[:start], variable)
                    values[f"expected_G_{suffix}"] = eos_value(
                        make_eos(method, parameters_),
                        "G",
                        pressure,
                        2000.0,
                        ratio * parameters_["V_0"],
                    )
                audit.scalars(path, offset + start, section, values)
            elif "dependent" in name or "depedent" in name:

                def evaluate(key, inputs):
                    selected = method
                    if key.endswith("_c"):
                        key, selected = key[:-2], "slb3-conductive"
                    if key == "G2":
                        selected = "slb2" if method == "slb3" else "mgd2"
                    phase = make_eos(selected, p)
                    if name in ["T-V dependent functions", "T-V dependent functions A"]:
                        temperature, ratio = inputs
                        pressure = 2e9
                    elif name == "P-T dependent functions":
                        pressure, temperature = inputs
                        ratio = 0.8
                    elif name == "Test P-V depedent":
                        pressure, ratio = inputs
                        temperature = 300.0
                    else:
                        pressure, temperature, ratio = inputs
                    return eos_value(
                        phase, key, pressure, temperature, ratio * p["V_0"]
                    )

                audit.maps(path, offset + start, section, evaluate)


MODIFIERS = {
    "LandauParams": ("landau_excesses", ["Tc_0", "V_D", "S_D"]),
    "LandauHPParams": ("landau_hp_excesses", ["T_0", "P_0", "Tc_0", "V_D", "S_D"]),
    "LandauSLB2022Params": ("landau_slb_2022_excesses", ["Tc_0", "V_D", "S_D"]),
    "LinearParams": ("linear_excesses", ["delta_V", "delta_S", "delta_E"]),
    "BraggWilliamsParams": (
        "bragg_williams_excesses",
        ["n", "factor", "Wh", "Wv", "deltaH", "deltaV"],
    ),
    "MagneticChsParams": (
        "magnetic_excesses_chs",
        ["structural_parameter", "tc", "tc_P", "moment", "moment_P"],
    ),
    "DebyeParams": ("debye_excesses", ["Cv_inf", "Theta_0"]),
    "DebyeDeltaParams": ("debye_delta_excesses", ["S_inf", "Theta_0"]),
    "EinsteinParams": ("einstein_excesses", ["Cv_inf", "Theta_0"]),
    "EinsteinDeltaParams": ("einstein_delta_excesses", ["S_inf", "Theta_0"]),
}


def modifier_params(kind, values):
    function, keys = MODIFIERS[kind]
    params = dict(zip(keys, values))
    if kind == "MagneticChsParams":
        params = dict(
            structural_parameter=params["structural_parameter"],
            curie_temperature=[params["tc"], params["tc_P"]],
            magnetic_moment=[params["moment"], params["moment_P"]],
        )
    return function, params


def audit_modifiers(audit):
    path = ROOT / "tests/eos/test_property_modifiers.cpp"
    text = path.read_text()
    for _, start, end in blocks(text, "test"):
        body = text[start:end]
        sections = blocks(body, "section") or [("", 0, len(body))]
        for name, section_start, section_end in sections:
            section = body[section_start:section_end]
            match = re.search(
                r"excesses::(\w+Params)\s+params\s*=\s*\{([^}]+)\}", section
            )
            if match is None:
                match = re.search(
                    r"excesses::(\w+Params)\s+params\s*=\s*\{([^}]+)\}",
                    body[:section_start],
                )
            function, params = modifier_params(match[1], numbers(match[2]))
            pressure, temperature = (
                (5e10, 2000.0) if name == "BW Low P" else (1e11, 1000.0)
            )
            expected = getattr(property_modifiers, function)(
                pressure, temperature, params
            )[0]
            values = {key + "_ref": value for key, value in expected.items()}
            # Keep the independent analytic Q=0 derivative check as well.
            if name == "BW Low P" and "double dGdP_ref = params.deltaV;" in section:
                del values["dGdP_ref"]
            audit.scalars(path, start + section_start, section, values)


def fixture_mineral(name):
    text = (ROOT / "tests/include/solution_fixtures.hpp").read_text()
    params = parameters(text, name + ".params")
    return make_eos("slb3", params)


def bridgmanite():
    pairs = [
        ("mg_si_perovskite", "[Mg][Si]O3"),
        ("fe_si_perovskite", "[Fe][Si]O3"),
        ("al_al_perovskite", "[Al][Al]O3"),
    ]
    return burnman.Solution(
        name="Bridgmanite",
        solution_model=solutionmodel.IdealSolution(
            [(fixture_mineral(name), sites) for name, sites in pairs]
        ),
        molar_fractions=[0.88, 0.07, 0.05],
    )


def pyrolite():
    wuestite = fixture_mineral("wuestite")
    wuestite.property_modifiers = [
        ["linear", dict(delta_V=1.0, delta_S=2.0, delta_E=3.0)]
    ]
    fp = burnman.Solution(
        name="Ferro-periclase",
        solution_model=solutionmodel.SymmetricRegularSolution(
            [(fixture_mineral("periclase"), "[Mg]O"), (wuestite, "[Fe]O")],
            energy_interaction=[[13000.0]],
        ),
        molar_fractions=[0.9, 0.1],
    )
    return burnman.Composite(
        [bridgmanite(), fp, fixture_mineral("ca_perovskite")], [0.7, 0.2, 0.1]
    )


def audit_materials(audit):
    short_names = dict(
        Vo="_molar_volume_unmodified",
        V="molar_volume",
        m="molar_mass",
        rho="density",
        E="molar_internal_energy",
        G="molar_gibbs",
        F="molar_helmholtz",
        S="molar_entropy",
        H="molar_enthalpy",
        KT="isothermal_bulk_modulus_reuss",
        KS="isentropic_bulk_modulus_reuss",
        invKT="isothermal_compressibility_reuss",
        invKS="isentropic_compressibility_reuss",
        mu="shear_modulus",
        gamma="grueneisen_parameter",
        alpha="thermal_expansivity",
        Cv="molar_heat_capacity_v",
        Cp="molar_heat_capacity_p",
        grad="isentropic_thermal_gradient",
        vp="p_wave_velocity",
        vphi="bulk_sound_velocity",
        vs="shear_wave_velocity",
    )
    for filename, factory, pressure, temperature, title in [
        ("mineral", None, 55e9, 1000.0, "Check py reference values"),
        ("solution", bridgmanite, 40e9, 2000.0, "Test reference values"),
        ("assemblage", pyrolite, 50e9, 2000.0, "Assemblage reference values"),
    ]:
        path = ROOT / f"tests/core/test_{filename}.cpp"
        text = path.read_text()
        case = next(case for case in blocks(text, "test") if case[0] == title)
        body = text[case[1] : case[2]]
        sections = (
            blocks(body, "section") if filename == "mineral" else [("", 0, len(body))]
        )
        for name, start, end in sections:
            phase = (
                factory()
                if factory
                else make_eos("mgd3", parameters(body[:start], "test_mineral.params"))
            )
            if name == "With excess":
                _, params = modifier_params(
                    "MagneticChsParams", [0.4, 800, 1e-8, 2.2, 1e-10]
                )
                phase.property_modifiers = [["magnetic_chs", params]]
            phase.set_state(pressure, temperature)
            section = body[start:end]
            values = {}
            for match in re.finditer(
                rf"(?:double|Eigen::Index)\s+(ref_\w+)\s*=\s*({NUMBER})\s*;", section
            ):
                prop = match[1].removeprefix("ref_")
                if filename == "mineral":
                    prop = short_names[prop]
                if filename == "assemblage" and prop == "shear_modulus":
                    prop = "effective_shear_modulus"
                values[match[1]] = (
                    len(phase.elements)
                    if prop == "n_elements"
                    else getattr(phase, prop)
                )
            audit.scalars(path, case[1] + start, section, values)
            arrays = {}
            for match in re.finditer(r"\b(ref_\w+)\s*<<", section):
                prop = match[1].removeprefix("ref_")
                # Composite has no compositional_basis property. Use the same
                # upstream basis completion used by Solution's property.
                if prop == "compositional_basis" and isinstance(
                    phase, burnman.Composite
                ):
                    arrays[match[1]] = complete_basis(phase.reaction_basis)[
                        phase.n_reactions :
                    ]
                else:
                    arrays[match[1]] = getattr(phase, prop)
            audit.arrays(path, case[1] + start, section, arrays)
            for match in re.finditer(
                r"std::vector<Eigen::Index>\s+(ref_\w+)\s*=\s*\{([^}]+)\}", section
            ):
                indices = getattr(phase, match[1].removeprefix("ref_"))
                literals = list(re.finditer(NUMBER, match[2]))
                if len(literals) != len(indices):
                    raise ValueError(f"Wrong index-list length: {path}: {match[1]}")
                for index, (literal, value) in enumerate(zip(literals, indices)):
                    audit.expect(
                        path,
                        case[1] + start + match.start(2) + literal.start(),
                        case[1] + start + match.start(2) + literal.end(),
                        value,
                        f"{match[1]}[{index}]",
                        as_integer=True,
                    )
            elements = re.search(
                r"std::vector<std::string>\s+ref_elements\s*=\s*\{([^}]+)\}", section
            )
            if elements and json.loads("[" + elements[1] + "]") != phase.elements:
                raise ValueError(
                    f"Reference element order differs from BurnMan: {path}"
                )


def audit_solution_models(audit):
    path = ROOT / "tests/core/test_solution_model.cpp"
    text = path.read_text()
    case = next(
        case for case in blocks(text, "test") if case[0] == "Reference value tests"
    )
    body = text[case[1] : case[2]]
    sites = [
        "[Mg]3[Al]2Si3O12",
        "[Fe]3[Al]2Si3O12",
        "[Ca]3[Al]2Si3O12",
        "[Ca]3[Fef]2Si3O12",
        "[Mg]3[Cr]2Si3O12",
    ]
    members = [(burnman.Mineral(), site) for site in sites]
    energy = [[4e3, 35e3, 91e3, 2e3], [4e3, 60e3, 6e3], [2e3, 47e3], [101e3]]
    volume = [
        [0.1e-5, 0.1e-5, 0.032e-5, 0],
        [0.1e-5, 0.032e-5, 0.01e-5],
        [0, 0.221e-5],
        [0.153e-5],
    ]
    entropy = [[0, 0, -1.7, 0], [0, -1.7, 0], [0, 33.8], [32.1]]
    models = dict(
        ideal=solutionmodel.IdealSolution(members),
        sym=solutionmodel.SymmetricRegularSolution(members, energy, volume, entropy),
        asym=solutionmodel.AsymmetricRegularSolution(
            members, [1, 1, 2.7, 0.4, 1.2], energy, volume, entropy
        ),
    )
    renames = dict(
        excess_gibbs_free_energy="excess_gibbs_energy",
        excess_partial_gibbs_free_energies="excess_partial_gibbs_energies",
    )
    for _, start, end in blocks(body, "section"):
        section = body[start:end]
        scalar_values, array_values = {}, {}
        for match in re.finditer(r"\b(ref_(ideal|sym|asym)_(\w+))\s*(?:=|\()", section):
            prop = renames.get(match[3], match[3])
            value = getattr(models[match[2]], prop)(
                2e9, 800.0, np.array([0.23, 0.17, 0.08, 0.34, 0.18])
            )
            (scalar_values if np.isscalar(value) else array_values)[match[1]] = value
        audit.scalars(path, case[1] + start, section, scalar_values)
        audit.arrays(path, case[1] + start, section, array_values)


def audit_averaging(audit):
    path = ROOT / "tests/tools/averaging/test_averaging_schemes.cpp"
    text = path.read_text()
    volumes = np.array([0.3, 0.45, 0.2, 0.05])
    bulk = np.array([120e9, 208.45e9, 264.2e9, 76.1e9])
    shear = np.array([96e9, 173.2e9, 207.35, 36.2])
    values = {}
    for key, cls in [
        ("voigt", averaging_schemes.Voigt),
        ("reuss", averaging_schemes.Reuss),
        ("vrh", averaging_schemes.VoigtReussHill),
    ]:
        values["ref_" + key] = cls().average_bulk_moduli(volumes, bulk, shear)
    for key, cls in [
        ("hsl", averaging_schemes.HashinShtrikmanLower),
        ("hsu", averaging_schemes.HashinShtrikmanUpper),
    ]:
        values[f"ref_{key}_K"] = cls().average_bulk_moduli(volumes, bulk, shear)
        values[f"ref_{key}_G"] = cls().average_shear_moduli(volumes, bulk, shear)
    audit.scalars(path, 0, text, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate stored numerical literals from Python BurnMan.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the detailed audit as JSON (usually under build/).",
    )
    args = parser.parse_args()
    source = verify_reference(burnman)
    audit = Audit()
    audit_components(audit)
    audit_eos(audit)
    audit_modifiers(audit)
    with tight_volumes():
        audit_materials(audit)
    audit_solution_models(audit)
    audit_averaging(audit)
    if "burnman_cpp" in sys.modules:
        raise RuntimeError(
            "The reference generator must not import the native implementation."
        )
    if args.report:
        args.report.write_text(
            json.dumps(dict(reference=REFERENCE, values=audit.records), indent=2) + "\n"
        )
    print(f"Reference import: {source / 'burnman/__init__.py'}")
    return 0 if audit.finish(args.write) else 1


if __name__ == "__main__":
    sys.exit(main())
