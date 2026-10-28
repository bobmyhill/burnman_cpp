#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Export the pinned Python pressure standards, wrappers and reference values.

The catalogue is checked in; neither native builds nor installed packages
import Python BurnMan. Regeneration requires the reference checkout, SciPy,
Black and clang-format.
Numeric expression translation is deliberately restricted and rejects unknown
syntax instead of silently generating an approximation. Tsuchiya's native
interpolator uses the reference's actual tensor-product cubic spline.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess

import numpy as np
import burnman
from burnman import calibrants

ROOT = Path(__file__).resolve().parents[1]
LICENSE = (
    "// SPDX-License-Identifier: GPL-3.0-or-later\n"
    "// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.\n"
)
PYLICENSE = LICENSE.replace("//", "#")


def literal(value):
    if isinstance(value, str):
        return json.dumps(value)
    return repr(float(value))


class Expression:
    def __init__(self, model="cold_vinet", differentiated=False):
        self.model = model
        self.math = "differentiation::" if differentiated else "std::"

    def __call__(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return literal(node.value)
        if isinstance(node, ast.Subscript):
            return f"{self(node.value)}.at({self(node.slice)})"
        if isinstance(node, ast.UnaryOp):
            assert isinstance(node.op, (ast.USub, ast.UAdd))
            return (
                f'({"-" if isinstance(node.op, ast.USub) else "+"}{self(node.operand)})'
            )
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return f"{self.math}pow({self(node.left)}, {self(node.right)})"
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[
                type(node.op)
            ]
            return f"({self(node.left)} {op} {self(node.right)})"
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            functions = {
                "np.power": self.math + "pow",
                "np.exp": self.math + "exp",
                "np.log": self.math + "log",
                "pressure_model.pressure": self.model,
                "do_cold_eos.pressure": "cold_vinet",
                "thermal_model._thermal_pressure": "thermal_pressure",
                "debye.thermal_energy": "thermal_energy",
                "pressure_fourth_order": "cold_bm4",
            }
            if name.startswith("_"):
                target = name
            else:
                target = functions[name]
            return f'{target}({", ".join(self(x) for x in node.args)})'
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def translate(function, name, differentiated=False):
    expression = Expression(differentiated=differentiated)
    scalar = "differentiation::Derivative" if differentiated else "double"
    arguments = [
        f"const Params &{a.arg}" if a.arg == "params" else f"{scalar} {a.arg}"
        for a in function.args.args
    ]
    used = {
        n.id
        for n in ast.walk(function)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    lines = [f'{scalar} {name}({", ".join(arguments)}) {{']
    for node in function.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0].id
            if target == "pressure_model":
                expression.model = {"BM3": "cold_bm3", "Vinet": "cold_vinet"}[
                    node.value.func.id
                ]
            elif target != "thermal_model":
                annotation = "" if target in used else "[[maybe_unused]] "
                lines.append(
                    f"  {annotation}const {scalar} {target} = {expression(node.value)};"
                )
        elif isinstance(node, ast.Return):
            lines.append(f"  return {expression(node.value)};")
        else:
            raise ValueError(f"Unsupported statement: {ast.dump(node)}")
    return "\n".join(lines + ["}"])


def spline_source(calibrant):
    spline = calibrant.interpolate_pressure._spline
    assert tuple(spline.k) == (3, 3)
    knots = ["{" + ", ".join(map(literal, k)) + "}" for k in spline.t]
    coefficients = ",\n".join(
        "    {" + ", ".join(map(literal, row)) + "}" for row in spline.c
    )
    return (
        """// Tensor-product cubic B-spline, including polynomial extrapolation.
template <std::size_t N>
double cubic(double x, const std::array<double, N + 4> &t,
             const std::array<double, N> &coefficients) {
  int span = 3;
  while (span < static_cast<int>(N) - 1 && x >= t[static_cast<std::size_t>(span + 1)])
    ++span;
  std::array<double, 4> d{};
  for (int j = 0; j < 4; ++j)
    d[static_cast<std::size_t>(j)] = coefficients[static_cast<std::size_t>(span - 3 + j)];
  for (int r = 1; r <= 3; ++r)
    for (int j = 3; j >= r; --j) {
      const auto i = static_cast<std::size_t>(span - 3 + j);
      const double alpha = (x - t[i]) / (t[i + static_cast<std::size_t>(4 - r)] - t[i]);
      d[static_cast<std::size_t>(j)] = (1. - alpha) * d[static_cast<std::size_t>(j - 1)]
                                       + alpha * d[static_cast<std::size_t>(j)];
    }
  return d[3];
}
double pressure_Tsuchiya_2003_Au(double volume, double temperature, const Params &params) {
  static constexpr std::array<double, 22> tx = @TX@;
  static constexpr std::array<double, 10> tt = @TT@;
  static constexpr std::array<std::array<double, 6>, 18> coefficients{{
@COEFFICIENTS@
  }};
  std::array<double, 18> along_volume{};
  for (std::size_t i = 0; i < 18; ++i)
    along_volume[i] = cubic<6>(temperature, tt, coefficients[i]);
  return cubic<18>(1. - volume / params.at("V_0"), tx, along_volume) * 1.e9;
}
""".replace("@TX@", knots[0])
        .replace("@TT@", knots[1])
        .replace("@COEFFICIENTS@", coefficients)
    )


def sources():
    source = Path(burnman.__file__).resolve().parent
    pin = json.loads((ROOT / "tests/reference/burnman.json").read_text())["commit"]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != pin:
        raise RuntimeError(f"Expected reference commit {pin}, got {commit}")
    changed = subprocess.run(
        ["git", "-C", str(source.parent), "diff", "--quiet", pin, "--", "burnman"],
        check=False,
    )
    if changed.returncode:
        raise RuntimeError("The reference BurnMan sources contain local changes.")
    modules = []
    for path in sorted((source / "calibrants").glob("*.py")):
        if path.stem in ("__init__", "tools"):
            continue
        committed = subprocess.check_output(
            [
                "git",
                "-C",
                str(source.parent),
                "show",
                f"{pin}:burnman/calibrants/{path.name}",
            ]
        )
        if path.read_bytes() != committed:
            raise RuntimeError(f"Reference calibrant source has local changes: {path}")
        module = getattr(calibrants, path.stem)
        classes = [
            (node, getattr(module, node.name))
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef)
        ]
        modules.append((path.stem, classes, ast.parse(path.read_text())))
    return pin, modules


def export():
    pin, modules = sources()
    header = [
        LICENSE,
        "#pragma once",
        '#include "burnman/core/calibrant.hpp"',
        "#include <vector>",
        "namespace burnman::calibrants {",
    ]
    cpp = [
        LICENSE,
        f"// Generated by tools/export_calibrants.py from BurnMan {pin}.",
        '#include "burnman/calibrants/catalogue.hpp"',
        '#include "models.hpp"',
        "#include <array>",
        "namespace burnman::calibrants {",
        "namespace {",
        "using namespace detail;",
        "constexpr double gas_constant = constants::physics::gas_constant;",
    ]
    bindings = [
        LICENSE,
        "#pragma once",
        '#include "burnman/calibrants/catalogue.hpp"',
        "namespace burnman::python {",
        "inline void bind_calibrant_catalogue(py::module_ &m) {",
    ]
    init = [
        PYLICENSE,
        '"""Published native pressure standards, in Python BurnMan module order."""',
        "",
    ]
    catalogue, factories = [], []
    functions = {}
    for module_name, classes, tree in modules:
        if module_name == "Dorogokupets_2007":
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name != "_pressure":
                    cpp.append(translate(node, node.name, differentiated=True))
            cpp.append(
                """// Exact volume derivative of the same published thermal energy.
double _pressure(double volume, double temperature, const Params &params) {
  const auto energy = _thermal_helmholtz_energy(
      differentiation::Derivative(volume, 1.), temperature, params);
  return cold_vinet(temperature, volume, params) - energy.derivative;
}"""
            )
        header.append(f"namespace {module_name} {{")
        bindings.append(f'  auto {module_name} = m.def_submodule("{module_name}");')
        wrapper = [
            PYLICENSE,
            f'"""Native pressure standards from {module_name}."""',
            "from ..classes.calibrant import Calibrant",
            "from .._core import calibrants as _native",
            "",
        ]
        init.append(f"from . import {module_name}")
        for node, cls in classes:
            instance = cls()
            name = node.name
            identifier = f"{module_name}.{name}"
            function_name = f"pressure_{module_name}_{name}"
            if module_name == "Dorogokupets_2007":
                function_name = "_pressure"
            elif module_name == "Tsuchiya_2003":
                cpp.append(spline_source(instance))
            else:
                constructor = next(
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                )
                function = next(
                    n for n in constructor.body if isinstance(n, ast.FunctionDef)
                )
                cpp.append(translate(function, function_name))
            parameter_literal = (
                "{"
                + ", ".join(
                    "{" + literal(k) + ", " + literal(v) + "}"
                    for k, v in sorted(instance.params.items())
                )
                + "}"
            )
            functions[identifier] = (function_name, parameter_literal)
            header.append(f"Calibrant {name}();")
            bindings.append(
                f'  {module_name}.def("{name}", &calibrants::{module_name}::{name});'
            )
            doc = ast.get_docstring(node) or f"{module_name} {name} pressure standard."
            wrapper += [
                f"class {name}(Calibrant):",
                "    " + repr(doc),
                "    def __init__(self):",
                f"        self._set_native(_native.{module_name}.{name}())",
                "",
            ]
            factories.append((identifier, f"{module_name}::{name}"))
            catalogue.append((identifier, instance))
        header.append("}")
        write(
            ROOT / f"python/burnman_cpp/calibrants/{module_name}.py", "\n".join(wrapper)
        )
    cpp.append("} // namespace")
    for identifier, factory in factories:
        function_name, parameters = functions[identifier]
        cpp.append(
            f'Calibrant {factory}() {{ return Calibrant({function_name}, "pressure", {parameters}); }}'
        )
    header += [
        "using Factory = Calibrant (*)();",
        "const std::vector<std::pair<std::string, Factory>> &catalogue();",
        "}",
    ]
    cpp += [
        "const std::vector<std::pair<std::string, Factory>> &catalogue() {",
        "  static const std::vector<std::pair<std::string, Factory>> entries{",
        ",\n".join(
            f'    {{"{identifier}", &{factory}}}' for identifier, factory in factories
        ),
        "  };",
        "  return entries;",
        "}",
        "} // namespace burnman::calibrants",
    ]
    bindings += ["}", "} // namespace burnman::python"]
    init += [
        "from . import tools",
        "",
        "__all__ = [",
        ",\n".join(f'    "{m[0]}"' for m in modules) + ",",
        '    "tools",',
        "]",
    ]
    write(ROOT / "include/burnman/calibrants/catalogue.hpp", "\n".join(header))
    write(ROOT / "src/calibrants/catalogue.cpp", "\n\n".join(cpp))
    write(ROOT / "python/bindings/calibrant_catalogue.hpp", "\n".join(bindings))
    write(ROOT / "python/burnman_cpp/calibrants/__init__.py", "\n".join(init))
    return pin, catalogue


def references(pin, catalogue):
    def difference(function, value, step):
        return (
            function(value - 2 * step)
            - 8 * function(value - step)
            + 8 * function(value + step)
            - function(value + 2 * step)
        ) / (12 * step)

    data = {"commit": pin, "calibrants": {}}
    for identifier, cal in catalogue:
        v0 = cal.params["V_0"]
        entries = {
            "params": cal.params,
            "pressure": [],
            "volume": [],
            "volume_outside_reference_domain": [],
        }
        differentiated = identifier.startswith("Dorogokupets_2007.")
        if differentiated:
            entries["stable_pressure"] = []
        for ratio in (1.02, 0.95, 0.85, 0.70):
            for temperature in (300.0, 1000.0, 2000.0):
                v = ratio * v0
                entries["pressure"].append(
                    [v, temperature, float(cal.pressure(v, temperature))]
                )
                if differentiated:
                    model = calibrants.Dorogokupets_2007
                    cold = model.do_cold_eos.pressure(temperature, v, cal.params)
                    derivative = difference(
                        lambda x: model._thermal_helmholtz_energy(
                            x, temperature, cal.params
                        ),
                        v,
                        v * 1.0e-4,
                    )
                    entries["stable_pressure"].append(
                        [v, temperature, float(cold - derivative)]
                    )
        for p in (0.0, 10.0e9, 30.0e9):
            for t in (300.0, 1000.0):
                try:
                    with np.errstate(invalid="ignore", over="ignore"):
                        value = float(cal.volume(p, t))
                    entries["volume"].append([p, t, value])
                except ValueError:
                    entries["volume_outside_reference_domain"].append([p, t])
        sv, st = v0 * 1.0e-4, 10.0
        VT = np.array([[sv**2, 0.2 * sv * st], [0.2 * sv * st, st**2]])
        p, C = cal.pressure(0.9 * v0, 1000.0, VT)
        entries["pressure_covariance"] = [
            0.9 * v0,
            1000.0,
            VT.tolist(),
            float(p),
            C.tolist(),
        ]
        PT = np.array([[1.0e16, 2.0e8], [2.0e8, 100.0]])
        v, C = cal.volume(10.0e9, 1000.0, PT)
        entries["volume_covariance"] = [
            10.0e9,
            1000.0,
            PT.tolist(),
            float(v),
            C.tolist(),
        ]
        if differentiated:
            # The reference's nested tiny finite differences are noisy. Use a
            # five-point derivative of its pressure function on wider probes
            # as an independent check on the native exact energy derivative.
            def gradient(volume, temperature):
                dpdv = difference(
                    lambda x: cal.pressure(x, temperature), volume, volume * 1.0e-3
                )
                dpdt = difference(lambda t: cal.pressure(volume, t), temperature, 1.0)
                return dpdv, dpdt

            dpdv, dpdt = gradient(0.9 * v0, 1000.0)
            J = np.array([[dpdv, dpdt], [1.0, 0.0], [0.0, 1.0]])
            entries["stable_pressure_covariance"] = (J @ VT @ J.T).tolist()
            dpdv, dpdt = gradient(float(v), 1000.0)
            J = np.array([[1.0 / dpdv, -dpdt / dpdv], [1.0, 0.0], [0.0, 1.0]])
            entries["stable_volume_covariance"] = (J @ PT @ J.T).tolist()
        data["calibrants"][identifier] = entries
    write(
        ROOT / "tests/reference/calibrants.json",
        json.dumps(data, indent=2, allow_nan=False),
    )


CHECK = False


def write(path, text):
    text = text.rstrip() + "\n"
    if path.suffix in (".cpp", ".hpp"):
        text = subprocess.check_output(
            ["clang-format", "--style=file", f"--assume-filename={path}"],
            input=text,
            text=True,
        )
    elif path.suffix == ".py":
        text = subprocess.check_output(
            ["black", "--quiet", "--stdin-filename", str(path), "-"],
            input=text,
            text=True,
        )
    if CHECK:
        actual = path.read_text() if path.exists() else ""
        if path.suffix == ".py" and actual:
            equal = ast.dump(ast.parse(actual)) == ast.dump(ast.parse(text))
        else:
            equal = actual == text
        if not equal:
            raise RuntimeError(f"Generated calibrant file differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--references", action="store_true", help="Also export Python reference values."
    )
    args = parser.parse_args()
    CHECK = args.check
    revision, entries = export()
    if args.references:
        references(revision, entries)
    print(
        f'{"Checked" if CHECK else "Exported"} {len(entries)} calibrants from {revision}.'
    )
