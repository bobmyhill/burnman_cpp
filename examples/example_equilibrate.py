# Adapted from BurnMan's examples/example_equilibrate.py.
# Copyright (C) 2012–2021 by the BurnMan team, GPL v2 or later.
# This adaptation is distributed under GPL v3 or later.
"""Reproduce the eight pure Python BurnMan equilibration examples in C++.

Install the wrapper and plotting dependencies with:
    python -m pip install -e '.[examples]'
    python examples/example_equilibrate.py

Mineral construction, solution models, state properties and equilibrium
solves run in C++. Python selects the scenarios and plots native results.
The example does not import or require the pure Python BurnMan package.
The default uses the original sampling and saves six figures. Use --show for
interactive plots, --case to select a scenario, or --quick for a smaller grid.
"""

import argparse
from pathlib import Path

import numpy as np
import burnman_cpp as bm
from burnman_cpp.minerals import HP11, JH15, SLB11


def assemblage(phases, fractions=None):
    if fractions is None:
        fractions = np.full(len(phases), 1.0 / len(phases))
    return bm.Assemblage(phases, fractions)


def phase_fraction(a, composition, index, fraction):
    prm = bm.get_equilibration_parameters(a, composition)
    return bm.PhaseFractionConstraint(index, fraction, prm)


def solve(composition, a, constraints, label, **kwargs):
    # Pressure, temperature and mole fractions need different absolute step
    # tolerances. A scalar 1e-3 Pa is below numerical precision at mantle P.
    kwargs.setdefault("tol", np.r_[1.0, 1.0e-6, np.full(a.n_endmembers, 1.0e-9)])
    result = bm.equilibrate(composition, a, constraints, **kwargs)
    solutions = result.sol_array.ravel()
    failures = [s for s in solutions if not s.success]
    print(
        f"  {label}: {len(solutions) - len(failures)}/{len(solutions)} converged",
        flush=True,
    )
    if failures:
        print(f"    First failure: {failures[0].message}", flush=True)
    if len(failures) == len(solutions):
        raise RuntimeError(f"{label}: no equilibrium found ({failures[0].message})")
    return result


def point(result):
    s = result.sol_array.item()
    if not s.success:
        raise RuntimeError(s.message)
    return s


def values(result):
    return np.array([s.x for s in result.sol_array.ravel() if s.success]).T


def describe(a):
    print(f"  {a.pressure / 1.e9:.4f} GPa, {a.temperature:.2f} K")
    for p, f in zip(a.phases, a.molar_fractions):
        suffix = (
            f", endmembers {np.array2string(p.molar_fractions, precision=4)}"
            if isinstance(p, bm.Solution)
            else ""
        )
        print(f"    {p.name}: {f:.4f}{suffix}")


class Example:
    def __init__(self, quick=False, plots=True):
        self.quick = quick
        self.plots = plots
        self.figures = {}

    def grid(self, lo, hi, count):
        return np.linspace(lo, hi, min(count, 5) if self.quick else count)

    def figure(self, name, **kwargs):
        if not self.plots:
            return None
        import matplotlib.pyplot as plt

        fig = plt.figure(**kwargs)
        self.figures[name] = fig
        return fig

    def aluminosilicates(self):
        sill, andal, ky = HP11.sill(), HP11.andalusite(), HP11.ky()
        comp = sill.formula
        a = assemblage([sill, andal, ky])
        result = solve(
            comp,
            a,
            [phase_fraction(a, comp, 2, 0.0), phase_fraction(a, comp, 0, 0.0)],
            "invariant",
        )
        p_inv, t_inv = point(result).x[:2]
        print(f"  Invariant point: {p_inv / 1.e9:.4f} GPa, {t_inv:.2f} K")
        fig = self.figure("aluminosilicates")
        ax = fig.subplots() if fig else None
        for pair, pressures in [
            ([andal, ky], self.grid(1.0e5, p_inv, 21)),
            ([sill, andal], self.grid(1.0e5, p_inv, 21)),
            ([sill, ky], self.grid(p_inv, 1.0e9, 21)),
        ]:
            a = assemblage(pair)
            r = solve(
                comp,
                a,
                [
                    [bm.PressureConstraint(p) for p in pressures],
                    phase_fraction(a, comp, 0, 0.0),
                ],
                f"{pair[0].name}/{pair[1].name}",
            )
            x = values(r)
            if ax:
                ax.plot(x[1], x[0] / 1.0e9)
        if ax:
            a = assemblage([sill, andal, ky])
            for p, t in [[0.2e9, 800.0], [0.6e9, 650.0], [0.5e9, 1000.0]]:
                a.set_state(p, t)
                ax.text(
                    t,
                    p / 1.0e9,
                    min(a.phases, key=lambda phase: phase.molar_gibbs).name,
                )
            ax.set(xlabel="Temperature (K)", ylabel="Pressure (GPa)")

    def ordering(self):
        comp = {"Mg": 1.0, "Fe": 1.0, "Si": 2.0, "O": 6.0}
        a = bm.simplify_composite_with_composition(
            assemblage([JH15.orthopyroxene()]), comp
        )
        opx = a.phases[0]
        opx.set_composition([1.0 / 3.0] * 3)
        a = assemblage([opx])
        temperatures = self.grid(2000.0, 300.0, 41)
        fig = self.figure("ordering")
        ax = fig.subplots() if fig else None
        for mg_number in np.linspace(10.0, 50.0, 5):
            comp.update(Mg=mg_number / 50.0, Fe=2.0 - mg_number / 50.0)
            r = solve(
                comp,
                a,
                [
                    bm.PressureConstraint(1.0e5),
                    [bm.TemperatureConstraint(t) for t in temperatures],
                ],
                f"Mg# {mg_number:g}",
            )
            names = r.prm.parameter_names
            candidates = [
                (i, -1.0 if "Derived member" in name else 1.0)
                for i, name in enumerate(names)
                if "ordered ferroenstatite" in name or "Derived member" in name
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"Cannot identify the ordering endmember: {names}")
            idx, sign = candidates[0]
            x = values(r)
            if ax:
                ax.plot(x[1], sign * x[idx], label=f"Mg# = {mg_number:g}")
        if ax:
            ax.set(
                xlabel="Temperature (K)", ylabel="Proportion of ordered orthopyroxene"
            )
            ax.legend()

    def gt_solvus(self):
        comp = {"Mg": 1.5, "Ca": 1.5, "Al": 2.0, "Si": 3.0, "O": 12.0}
        reduced = bm.simplify_composite_with_composition(
            assemblage([SLB11.garnet(), SLB11.garnet()]), comp
        )
        gt1, gt2 = reduced.phases
        gt1.set_composition([0.05, 0.95])
        gt2.set_composition([0.95, 0.05])
        a = assemblage([gt1, gt2], [0.5, 0.5])
        r = solve(
            comp,
            a,
            [
                bm.PressureConstraint(1.0e5),
                [bm.TemperatureConstraint(t) for t in self.grid(300.0, 601.4, 301)],
            ],
            "pyrope–grossular solvus",
        )
        solutions = list(r.sol_array.flat)
        for i, s in enumerate(solutions):
            if not s.success:
                # The branches merge at the critical point, where the split
                # between identical phases is undetermined. The continuation
                # predictor can become singular; retry with distinct, nearby
                # compositions rather than dropping the endpoint from the plot.
                gt1.set_composition([0.48, 0.52])
                gt2.set_composition([0.52, 0.48])
                a.set_fractions([0.5, 0.5])
                retry = solve(
                    comp,
                    a,
                    [bm.PressureConstraint(1.0e5), bm.TemperatureConstraint(s.x[1])],
                    "solvus endpoint retry",
                )
                solutions[i] = point(retry)
        x = np.array([s.x for s in solutions if s.success]).T
        fig = self.figure("gt_solvus")
        if fig:
            ax = fig.subplots()
            ax.plot(1.0 - x[3], x[1])
            ax.plot(1.0 - x[5], x[1])
            ax.text(0.5, 400.0, "miscibility gap", ha="center")
            ax.set(xlabel="Molar proportion of pyrope", ylabel="Temperature (K)")

    def fper_ol(self):
        ol, fper = SLB11.mg_fe_olivine(), SLB11.ferropericlase()
        ol.set_composition([0.93, 0.07])
        fper.set_composition([0.9, 0.1])
        a = assemblage([ol, fper], [0.7, 0.3])
        a.set_state(0.0, 1000.0)
        comp = {"Mg": 1.0, "Fe": 0.5, "Si": 0.5, "O": 2.5}
        r = solve(
            comp,
            a,
            [
                [bm.PressureConstraint(p) for p in np.linspace(0.0, 10.0e9, 3)],
                [bm.TemperatureConstraint(t) for t in self.grid(1000.0, 1500.0, 11)],
            ],
            "olivine/ferropericlase partitioning",
        )
        fig = self.figure("fper_ol")
        if fig:
            ax = fig.subplots()
            for row in r.sol_array:
                x = np.array([s.x for s in row if s.success]).T
                (line,) = ax.plot(x[1], x[3], label=f"p(fa), {x[0, 0]/1.e9:g} GPa")
                ax.plot(
                    x[1],
                    x[5],
                    color=line.get_color(),
                    ls="--",
                    label=f"p(wus), {x[0, 0]/1.e9:g} GPa",
                )
            ax.set(xlabel="Temperature (K)", ylabel="Proportion (mol fraction)")
            ax.legend()

    def fixed_ol_composition(self):
        ol, wad = SLB11.mg_fe_olivine(), SLB11.mg_fe_wadsleyite()
        ol.set_composition([0.5, 0.5])
        wad.set_composition([0.6, 0.4])
        a = assemblage([ol, wad], [0.7, 0.3])
        a.set_state(10.0e9, 1200.0)
        comp = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        prm = bm.get_equilibration_parameters(a, comp)
        constraint = bm.PhaseCompositionConstraint(
            0, ["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45, a, prm
        )
        r = solve(
            comp,
            a,
            [bm.PressureConstraint(10.0e9), constraint],
            "fixed olivine composition",
        )
        bm.set_composition_and_state_from_parameters(a, point(r).x)
        describe(a)

    def upper_mantle(self):
        ol, opx, gt = SLB11.mg_fe_olivine(), SLB11.orthopyroxene(), SLB11.garnet()
        ol.set_composition([0.93, 0.07])
        opx.set_composition([0.8, 0.1, 0.05, 0.05])
        gt.set_composition([0.8, 0.1, 0.05, 0.03, 0.02])
        a = assemblage([ol, opx, gt], [0.7, 0.1, 0.2])
        a.set_state(10.0e9, 1500.0)
        comp = {
            "Na": 0.02,
            "Fe": 0.2,
            "Mg": 2.0,
            "Si": 1.9,
            "Ca": 0.2,
            "Al": 0.4,
            "O": 6.81,
        }
        r = solve(
            comp,
            a,
            [bm.PressureConstraint(10.0e9), bm.TemperatureConstraint(1500.0)],
            "upper mantle",
            max_iterations=20,
        )
        bm.set_composition_and_state_from_parameters(a, point(r).x)
        describe(a)

    def lower_mantle(self):
        bdg, ppv, fper, cpv = (
            SLB11.mg_fe_bridgmanite(),
            SLB11.post_perovskite(),
            SLB11.ferropericlase(),
            SLB11.ca_perovskite(),
        )
        for p, name in zip([bdg, ppv, fper, cpv], ["bdg", "ppv", "fper", "cpv"]):
            p.name = name
        bdg.set_composition([0.86, 0.1, 0.04])
        ppv.set_composition([0.86, 0.1, 0.04])
        fper.set_composition([0.9, 0.1])
        comp = {"Fe": 0.2, "Mg": 2.0, "Si": 1.9, "Ca": 0.2, "Al": 0.4, "O": 6.8}
        a = assemblage([bdg, fper, cpv])
        r = solve(
            comp,
            a,
            [bm.PressureConstraint(25.0e9), bm.TemperatureConstraint(1600.0)],
            "reference entropy",
        )
        bm.set_composition_and_state_from_parameters(a, point(r).x)
        entropy = (
            a.n_moles * a.molar_entropy
        )  # Constraint fixes total entropy of the specified bulk.
        a = assemblage([bdg, fper, ppv, cpv])
        a.set_state(*point(r).x[:2])
        r = solve(
            comp,
            a,
            [bm.EntropyConstraint(entropy), phase_fraction(a, comp, 2, 0.0)],
            "post-perovskite in",
        )
        p_in, t_in = point(r).x[:2]
        bm.set_composition_and_state_from_parameters(a, point(r).x)
        r = solve(
            comp,
            a,
            [bm.EntropyConstraint(entropy), phase_fraction(a, comp, 0, 0.0)],
            "bridgmanite out",
        )
        p_out, t_out = point(r).x[:2]
        print(
            f"  ppv in: {p_in/1.e9:.3f} GPa, {t_in:.2f} K; bdg out: {p_out/1.e9:.3f} GPa, {t_out:.2f} K"
        )
        fig = self.figure("lower_mantle", figsize=(16, 8))
        axes = fig.subplots(2, 3).ravel() if fig else None
        for j, (pressures, phases) in enumerate(
            [
                (self.grid(25.0e9, p_in, 21), [bdg, fper, cpv]),
                (self.grid(p_in, p_out, 21), [bdg, fper, ppv, cpv]),
                (self.grid(p_out, 140.0e9, 21), [ppv, fper, cpv]),
            ]
        ):
            a = assemblage(phases)
            r = solve(
                comp,
                a,
                [
                    [bm.PressureConstraint(p) for p in pressures],
                    bm.EntropyConstraint(entropy),
                ],
                f"isentrope segment {j+1}",
            )
            if axes is None:
                continue
            x, names = values(r), r.prm.parameter_names
            pressures = x[0] / 1.0e9
            axes[0].plot(pressures, x[1], color="black")
            for i, (p, color) in enumerate(
                zip([bdg, fper, cpv, ppv], ["red", "green", "blue", "orange"])
            ):
                if f"x({p.name})" in names:
                    axes[i + 1].plot(
                        pressures, x[names.index(f"x({p.name})")], color=color
                    )
                axes[i + 1].set(
                    xlim=(0, 140), ylim=(0, 2), ylabel=f"n moles of {p.name}"
                )
            per_idx = names.index("x(fper)")
            for p, color in [(bdg, "red"), (ppv, "blue")]:
                if f"x({p.name})" in names:
                    idx = names.index(f"x({p.name})")
                    kd = (
                        x[idx + 1]
                        * (1.0 - x[per_idx + 1])
                        / ((1.0 - x[idx + 1] - x[idx + 2]) * x[per_idx + 1])
                    )
                    axes[5].plot(
                        pressures,
                        kd,
                        color=color,
                        label=f"{p.name} K$_D$" if j == 1 else None,
                    )
        if axes is not None:
            for ax in axes:
                ax.set_xlabel("Pressure (GPa)")
            axes[0].set_ylabel("Temperature (K)")
            axes[5].set(ylim=(0, 1), ylabel="[FeSiO$_3$/MgSiO$_3$]/[FeO/MgO]")
            axes[5].legend()

    def olivine_polymorphs(self):
        ol, wad, rw = (
            SLB11.mg_fe_olivine(),
            SLB11.mg_fe_wadsleyite(),
            SLB11.mg_fe_ringwoodite(),
        )
        for p, f in zip([ol, wad, rw], [[0.93, 0.07], [0.91, 0.09], [0.93, 0.07]]):
            p.set_composition(f)
        comp = {"Fe": 0.2, "Mg": 1.8, "Si": 1.0, "O": 4.0}
        a = assemblage([ol, wad, rw], [1.0, 0.0, 0.0])
        r = solve(
            comp,
            a,
            [phase_fraction(a, comp, 0, 0.0), phase_fraction(a, comp, 2, 0.0)],
            "ol–wad–rw invariant 1",
        )
        p_inv1, t_inv1 = point(r).x[:2]
        a = assemblage([ol, wad, rw])
        r = solve(
            comp,
            a,
            [phase_fraction(a, comp, 1, 0.0), phase_fraction(a, comp, 2, 0.0)],
            "ol–wad–rw invariant 2",
        )
        p_inv2, t_inv2 = point(r).x[:2]
        print(
            f"  Invariants: ({p_inv1/1.e9:.3f} GPa, {t_inv1:.2f} K), ({p_inv2/1.e9:.3f} GPa, {t_inv2:.2f} K)"
        )
        fig = self.figure("olivine_polymorphs")
        ax = fig.subplots() if fig else None
        t0, t1 = 573.15, 1773.15
        curves = [
            (t0, t_inv1, [ol, wad, rw], 0, 0.0),
            (t0, t_inv2, [ol, wad, rw], 1, 0.0),
            (t_inv2, t_inv1, [ol, wad, rw], 2, 0.0),
            (t0, t_inv2, [ol, rw], 1, 0.0),
            (t_inv2, t1, [ol, wad], 1, 0.0),
            (t_inv1, t1, [ol, wad], 1, 1.0),
            (t_inv1, t1, [wad, rw], 1, 0.0),
            (t0, t1, [wad, rw], 1, 1.0),
        ]
        for i, (lo, hi, phases, index, fraction) in enumerate(curves):
            a = assemblage(phases, [0.0, 1.0] if i == 7 else None)
            r = solve(
                comp,
                a,
                [
                    [bm.TemperatureConstraint(t) for t in self.grid(lo, hi, 8)],
                    phase_fraction(a, comp, index, fraction),
                ],
                f"polymorph boundary {i+1}",
            )
            if ax:
                x = values(r)
                ax.plot(x[1], x[0] / 1.0e9, color="black")
        rw2, bdg, fper = (
            SLB11.mg_fe_ringwoodite(),
            SLB11.mg_fe_bridgmanite(),
            SLB11.ferropericlase(),
        )
        rw2.set_composition(rw.molar_fractions)
        fper.set_composition([0.9, 0.1])
        a = bm.simplify_composite_with_composition(assemblage([rw2, bdg, fper]), comp)
        a.phases[1].set_composition([0.86, 0.14])
        for fraction in [0.0, 1.0]:
            r = solve(
                comp,
                a,
                [
                    [bm.TemperatureConstraint(t) for t in self.grid(t0, t1, 8)],
                    phase_fraction(a, comp, 0, fraction),
                ],
                f"ringwoodite breakdown, fraction {fraction:g}",
            )
            if ax:
                x = values(r)
                ax.plot(x[1], x[0] / 1.0e9, color="black")
        if ax:
            for t, p, name in [
                (1750.0, 10.0, "olivine"),
                (1750.0, 15.0, "wadsleyite"),
                (1750.0, 20.0, "ringwoodite"),
                (1750.0, 24.0, "bridgmanite + ferropericlase"),
            ]:
                ax.text(t, p, name, ha="right")
            ax.set(xlabel="Temperature (K)", ylabel="Pressure (GPa)", ylim=(5, 25))


CASES = (
    "aluminosilicates",
    "ordering",
    "gt_solvus",
    "fper_ol",
    "fixed_ol_composition",
    "upper_mantle",
    "lower_mantle",
    "olivine_polymorphs",
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer samples, retaining all scenarios.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run calculations without loading matplotlib.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display figures after saving them."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/example_equilibrate")
    )
    args = parser.parse_args()
    if not args.no_plots and not args.show:
        import matplotlib

        matplotlib.use("Agg")
    example = Example(quick=args.quick, plots=not args.no_plots)
    for name in CASES if args.case == "all" else (args.case,):
        print(f"\n{name}", flush=True)
        getattr(example, name)()
    if example.figures:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in example.figures.items():
            fig.tight_layout()
            path = args.output_dir / f"{name}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        import matplotlib.pyplot as plt

        if args.show:
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
