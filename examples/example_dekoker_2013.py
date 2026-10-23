# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
"""MgO-SiO2 mixing and melting with the native DKS equations of state.

Run: python examples/example_dekoker_2013.py (requires NumPy, SciPy, Matplotlib).
Fits two excess-Gibbs coefficients to seven intermediate liquids at each P/T,
with lambda=1.43, fixed pure terms and unchanged catalogue parameters. All
mixing quantities are per mole of MgO + SiO2. This illustrative least-squares
model differs from the authors' unavailable continuous interaction functions.
Source: de Koker et al. (2013), doi:10.1016/j.epsl.2012.11.026.
"""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, newton
from scipy.special import xlogy

import burnman_cpp as bm
from burnman_cpp.minerals import DKS_2013_liquids as liquids, DKS_2013_solids as solids

SHOW_PLOTS = False
OUTPUT_DIR = Path(__file__).with_name("dekoker_2013_output")
R, ASYMMETRY = bm.constants.gas_constant, 1.43
COMPOSITIONS = (
    "MgO Mg5SiO7 Mg2SiO4 Mg3Si2O7 MgSiO3 MgSi2O5 MgSi3O7 MgSi5O11 SiO2".split()
)
STATES = [
    (0, 3000),
    (5, 3000),
    (10, 3000),
    (25, 3000),
    (50, 4000),
    (75, 5000),
    (100, 6000),
    (135, 6000),
]


def mixing_basis(x, derivative=False):
    denominator = ASYMMETRY + (1 - ASYMMETRY) * np.asarray(x)
    y = x / denominator
    if derivative:
        return (
            np.stack((2 * y - 3 * y * y, 1 - 4 * y + 3 * y * y), axis=-1)
            * (ASYMMETRY / denominator**2)[..., None]
        )
    return np.stack((y * y * (1 - y), y * (1 - y) ** 2), axis=-1)


def ideal_entropy(x):
    y = np.asarray(x) / (ASYMMETRY + (1 - ASYMMETRY) * np.asarray(x))
    return -R * (xlogy(y, y) + xlogy(1 - y, 1 - y))


@lru_cache(maxsize=32)
def seifertite_correction(pressure):
    """Static Vinet volume difference integrated from 110 GPa; neglect thermal terms.

    Oganov et al. (2005), doi:10.1103/PhysRevB.71.064104, Table II.
    """
    if pressure <= 110e9:
        return 0.0
    eos = bm.make_eos("vinet")
    params = [
        bm.mineral_params(
            V_0=v * 1e-30 * bm.constants.Avogadro, K_0=k * 1e9, Kprime_0=kp
        )
        for v, k, kp in ((23.1559, 318.326, 4.3726), (22.7805, 324.426, 4.2327))
    ]
    for p in params:
        eos.validate_parameters(p)
    return quad(
        lambda p: eos.compute_volume(p, 300, params[1])
        - eos.compute_volume(p, 300, params[0]),
        110e9,
        pressure,
    )[0]


class LeastSquaresLiquid:
    def __init__(self, use_seifertite=True):
        self.liquids = [getattr(liquids, name + "_liquid")() for name in COMPOSITIONS]
        self.solids = [solids.periclase(), solids.perovskite(), solids.stishovite()]
        self.oxides = np.array(
            [m.formula.get("Mg", 0) + m.formula.get("Si", 0) for m in self.liquids]
        )
        self.x = np.array([m.formula.get("Si", 0) for m in self.liquids]) / self.oxides
        self.projection = np.linalg.pinv(mixing_basis(self.x))
        self.use_seifertite = use_seifertite

    @lru_cache(maxsize=4096)
    def state(self, pressure, temperature):
        for m in self.liquids:
            m.set_state(pressure, temperature)
        values = (
            np.array(
                [
                    [m.molar_gibbs, m.molar_enthalpy, m.molar_entropy, m.molar_volume]
                    for m in self.liquids
                ]
            )
            / self.oxides[:, None]
        )
        mixing = (
            values - (1 - self.x[:, None]) * values[0] - self.x[:, None] * values[-1]
        )
        excess = mixing.copy()
        excess[:, 2] -= ideal_entropy(self.x)
        weights = self.projection @ excess
        weights[:, 0] = weights[:, 1] - temperature * weights[:, 2]
        return values, mixing, weights  # Columns: G, H, S, V.

    def gibbs(self, x, pressure, temperature):
        values, _, weights = self.state(pressure, temperature)
        return (
            (1 - x) * values[0, 0]
            + x * values[-1, 0]
            - temperature * ideal_entropy(x)
            + mixing_basis(x) @ weights[:, 0]
        )

    def chemical_potentials(self, x, pressure, temperature):
        values, _, weights = self.state(pressure, temperature)
        denominator = ASYMMETRY + (1 - ASYMMETRY) * x
        y = x / denominator
        f = xlogy(y, y) + xlogy(1 - y, 1 - y)
        df = ASYMMETRY / denominator**2 * np.log(max(y, 1e-300) / max(1 - y, 1e-300))
        excess = mixing_basis(x) @ weights[:, 0]
        slope = mixing_basis(x, derivative=True) @ weights[:, 0]
        return (
            values[[0, -1], 0]
            + R * temperature * (f + np.array([-x, 1 - x]) * df)
            + (excess + np.array([-x, 1 - x]) * slope)
        )

    def saturation_residuals(self, x, pressure, temperature, mg_number=1.0):
        mu_mg, mu_si = self.chemical_potentials(x, pressure, temperature)
        for m in self.solids:
            m.set_state(pressure, temperature)
        g = np.array([m.molar_gibbs for m in self.solids])
        if self.use_seifertite:
            g[2] += seifertite_correction(pressure)
        # Ideal Mg-Fe activities: K(Lq-Fp)=0.4 and K(Pv-Fp)=1.
        g[:2] -= R * temperature * np.log(mg_number + 0.4 * (1 - mg_number))
        return np.array([mu_mg, mu_mg + mu_si, mu_si]) - g

    def eutectic(self, pressure, pair, guess, mg_number=1.0):
        def residual(xt):
            return (
                self.saturation_residuals(xt[0], pressure, xt[1], mg_number)[list(pair)]
                / 1e5
            )

        result = least_squares(
            residual,
            guess,
            bounds=([0.001, 1500], [0.999, 9500]),
            x_scale=[0.2, 3000],
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        error = np.max(np.abs(residual(result.x))) * 1e5
        if not result.success or error > 0.01:
            raise RuntimeError(f"Eutectic solve failed: {error:g} J/mol")
        return result.x, error

    def liquidus(self, pressure, phase, xs, guess, mg_number=1.0):
        temperatures = []
        for x in xs:

            def residual(t):
                return self.saturation_residuals(
                    float(x), pressure, float(t), mg_number
                )[phase]

            guess = newton(residual, guess, x1=guess + 10, tol=1e-6, maxiter=40)
            if abs(residual(guess)) > 0.01:
                raise RuntimeError("Liquidus residual exceeds 0.01 J/mol")
            temperatures.append(guess)
        return temperatures


def mixing_figure(model):
    grid = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True, layout="constrained")
    for (p, t), color in zip(STATES, plt.cm.turbo(np.linspace(0, 1, len(STATES)))):
        _, mixing, weights = model.state(p * 1e9, t)
        for ax, column, scale in zip(axes, (1, 3, 2), (1e-3, 1e6, 1)):
            curve = mixing_basis(grid) @ weights[:, column]
            if column == 2:
                curve += ideal_entropy(grid)
            ax.plot(grid, curve * scale, color=color, label=f"{p} GPa; {t} K")
            ax.plot(model.x[1:-1], mixing[1:-1, column] * scale, "o", color=color, ms=3)
    for ax, label in zip(axes, ("H (kJ/mol)", "V (cm³/mol)", "S (J/mol/K)")):
        ax.set(ylabel="Mixing " + label, xlim=(0, 1))
        ax.axhline(0, color=".8", lw=0.5)
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel(r"$X(\mathrm{SiO}_2)$")
    fig.suptitle("Least-squares liquid mixing; λ = 1.43")
    return fig


def phase_diagram(model):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
    # The lower isobar uses 25 GPa (the paper labels it 24 GPa).
    for ax, p_gpa in zip(axes, (25, 136)):
        p, t = p_gpa * 1e9, 2800 if p_gpa == 25 else 4900
        for mg, color in ((1.0, "tab:red"), (0.9, "tab:green"), (0.8, "tab:blue")):
            left, _ = model.eutectic(p, (0, 1), [0.41, t], mg)
            right, _ = model.eutectic(p, (1, 2), [0.68, t - 200], mg)
            for phase, start, end in (
                (0, left, 0),
                (1, left, 0.5),
                (1, right, 0.5),
                (2, right, 1),
            ):
                xs = np.linspace(start[0], end, 41)
                ax.plot(xs, model.liquidus(p, phase, xs, start[1], mg), color=color)
            for start, endpoints in ((left, [0, 0.5]), (right, [0.5, 1])):
                ax.plot(endpoints, [start[1]] * 2, color=color, lw=0.7)
                ax.plot(*start, "o", color=color, ms=3)
            ax.plot([], [], color=color, label=f"Mg# {100*mg:g}")
        ax.set(title=f"{p_gpa} GPa", xlim=(0, 1), xlabel=r"$X(\mathrm{SiO}_2)$")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Temperature (K)")
    fig.suptitle("Least-squares melting example (approximate seifertite above 110 GPa)")
    return fig


def main():
    model = LeastSquaresLiquid()
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, plot in (
        ("figure_1_least_squares", mixing_figure),
        ("figure_2_least_squares", phase_diagram),
    ):
        fig = plot(model)
        for extension in ("pdf", "png"):
            fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=200)
    print(f"Saved figures to {OUTPUT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
