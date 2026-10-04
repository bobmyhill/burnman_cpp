import pytest

import burnman_cpp as bm


def oxide_params(element="Mg", energy_shift=0.0):
    """SLB3 endmembers with matching elastic properties and variable free energy."""
    return {
        "name": f"{element}O",
        "formula": {element: 1.0, "O": 1.0},
        "equation_of_state": "slb3",
        "molar_mass": 0.0403044 if element == "Mg" else 0.0718444,
        "n": 2,
        "F_0": -569444.6 + energy_shift,
        "V_0": 1.1244e-5,
        "K_0": 1.613836e11,
        "Kprime_0": 3.84045,
        "G_0": 1.309e11,
        "Gprime_0": 2.1438,
        "Debye_0": 767.0977,
        "grueneisen_0": 1.36127,
        "q_0": 1.7217,
        "eta_s_0": 2.81765,
    }


def make_model(kind=bm.IdealSolution, shifts=(0.0, 0.0), **kwargs):
    endmembers = [
        (bm.Mineral(oxide_params("Mg", shifts[0])), "[Mg]O"),
        (bm.Mineral(oxide_params("Fe", shifts[1])), "[Fe]O"),
    ]
    return kind(endmembers, **kwargs)


@pytest.fixture
def endmembers():
    return [
        (bm.Mineral(oxide_params("Mg")), "[Mg]O"),
        (bm.Mineral(oxide_params("Fe")), "[Fe]O"),
    ]


@pytest.fixture
def solution():
    return bm.Solution(make_model(), [0.8, 0.2], name="oxide")


@pytest.fixture
def two_phase_assemblage():
    a = bm.Solution(make_model(shifts=(0.0, 10000.0)), [0.8, 0.2], name="Mg-rich")
    b = bm.Solution(make_model(shifts=(10000.0, 0.0)), [0.2, 0.8], name="Fe-rich")
    assemblage = bm.Assemblage([a, b], [0.4, 0.6])
    assemblage.set_state(1.0e9, 2000.0)
    return assemblage
