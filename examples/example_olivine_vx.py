#!/usr/bin/env python3
"""Olivine V-X diagram at 1673 K using the shared olivine example.

Both endpoints contain one mole of formula units. The closed-system total
volume spans 36-45 cm³; pressure is solved, including volume changes through
coexistence of olivine, wadsleyite and ringwoodite (SLB2011).

Run: python examples/example_olivine_vx.py
"""

from functools import partial

from example_olivine_px import calculate as _calculate, main as _main

calculate = partial(_calculate, diagram="VX")
main = partial(_main, diagram="VX")

if __name__ == "__main__":
    main()
