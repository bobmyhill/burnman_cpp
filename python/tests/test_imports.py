"""Importing the native package does not initialize optional numerical tools."""

import os
from pathlib import Path
import subprocess
import sys

import burnman_cpp as bm
import pytest


@pytest.mark.parametrize(
    "operation",
    [
        "m = bm.minerals.SLB_2011.forsterite(); m.set_state(1e9, 300.); assert m.density > 0.",
        "from burnman_cpp.utils.chemistry import dictionarize_formula; assert dictionarize_formula('MgO') == {'Mg': 1., 'O': 1.}",
        "from burnman_cpp.tools import eos; assert callable(eos.check_eos_consistency)",
    ],
)
def test_lightweight_import_in_fresh_process(operation):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(bm.__file__).resolve().parent.parent), *sys.path]
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import burnman_cpp as bm; "
            + operation
            + "; assert not {'scipy', 'sympy', 'matplotlib'} & sys.modules.keys()",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_lazy_exports_retain_public_names_and_identity():
    from importlib import import_module

    tools = bm.tools
    for name in tools.__all__:
        assert name in dir(tools)
        value = getattr(tools, name)
        assert getattr(tools, name) is value
        if name in bm._LAZY_TOOLS:
            assert getattr(bm, name) is value
    for name in tools._EXPORTS:
        assert getattr(tools, name) is import_module(f"burnman_cpp.tools.{name}")
    for name in bm.__all__:
        assert name in dir(bm)
        getattr(bm, name)
    with pytest.raises(AttributeError):
        getattr(bm, "unknown_burnman_export")
    with pytest.raises(AttributeError):
        getattr(tools, "unknown_burnman_export")
