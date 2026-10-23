"""Nine unmodified liquid EOSs against independently transcribed Table 1."""

from pathlib import Path
import json
import runpy

import pytest

import burnman_cpp as bm

BENCHMARK = Path(__file__).parents[2] / "benchmarks/dekoker_2013/benchmark.py"
REFERENCE = json.loads((BENCHMARK.parent / "reference/endmembers.json").read_text())


@pytest.fixture(scope="module")
def benchmark():
    pytest.importorskip("matplotlib")
    return runpy.run_path(str(BENCHMARK))


@pytest.mark.parametrize(
    "row", REFERENCE["compositions"], ids=lambda r: r["composition"]
)
def test_liquid_endmember_against_paper_table_1(benchmark, row):
    phase = getattr(bm.minerals.DKS_2013_liquids, row["composition"] + "_liquid")()
    native = benchmark["native_table_1"](phase)
    for name, calculated in native.items():
        paper = row["table_1"][name]
        # Preserve the catalogue's known small differences from the printed
        # bulk moduli; do not adjust EOS parameters to force a Table 1 match.
        known_bulk_residual = {
            "MgO": 0.06607264466041585,
            "Mg5SiO7": 0.033552695998746884,
        }
        if name == "K_T_GPa" and row["composition"] in known_bulk_residual:
            assert calculated - paper["value"] == pytest.approx(
                known_bulk_residual[row["composition"]], abs=1e-7
            )
            continue
        assert abs(calculated - paper["value"]) <= paper["printed_uncertainty"], name
