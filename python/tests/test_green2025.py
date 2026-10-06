"""Independent assemblage landmarks in Green et al. (2025), Supplement 3.

The reference is the published MAGEMin figures, not Python BurnMan output.
The six examples retain the oxide amounts and manually read field labels.
"""

from collections import Counter
import importlib
from pathlib import Path

import pytest
import burnman_cpp as bm

NAMES = ("klb1", "re46", "at1", "mix1g", "ton101u", "ton101s")


@pytest.fixture(scope="module", params=NAMES)
def benchmark(request):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
        example = importlib.import_module(f"example_green2025_{request.param}")
        runner = importlib.import_module("green2025")
    composition = bm.Composition(example.OXIDES, unit_type="molar")
    return example, composition, runner.candidate_phases(composition)


def test_published_interior_assemblages(benchmark):
    example, composition, phases = benchmark
    for temperature_c, pressure_kbar, expected in example.REFERENCE_POINTS:
        state = bm.stable_equilibrium(
            composition.atomic_composition,
            phases,
            pressure_kbar * 1.0e8,
            temperature_c + 273.15,
        )
        assert state.success, state.message
        assert state.mass_balance_error < 1.0e-8
        assert state.minimum_affinity >= -0.2
        actual = Counter(p.name.split(" #")[0] for p in state.phases)
        assert actual == Counter(expected.split()), (temperature_c, pressure_kbar)
