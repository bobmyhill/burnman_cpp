# Nine DKS liquid-composition comparisons

Compare the unmodified native `dks_l` models for **MgO, Mg5SiO7, Mg2SiO4,
Mg3Si2O7, MgSiO3, MgSi2O5, MgSi3O7, MgSi5O11 and SiO2** with de Koker,
Karki & Stixrude (2013).

```sh
python benchmarks/dekoker_2013/benchmark.py
```

The benchmark contains only individual-composition EOS comparisons:

- Nine P–V/E–V overlays on the original supplement Figures S2–S10.
- Six Table 1 properties for each composition at 0 GPa, 3000 K: molar volume,
  bulk modulus, entropy, Cv, Cp and Grüneisen parameter.

Outputs are saved in `results/`: `endmember_comparisons.pdf` contains all nine
comparisons, with separate `<composition>_comparison.pdf/png` files, Table 1
CSV/JSON and sampled isotherms. Units are per mole of the named formula;
Cv and Cp are normalized by the number of atoms times R. Energy curves omit
volumes where the native stable volume solve does not return the same branch.
The overlays are visual comparisons; the Table 1 report records quantitative
differences and the paper's printed uncertainties.

The catalogue bulk moduli of MgO and Mg5SiO7 exceed Table 1 by 0.0661 and
0.0336 GPa (1.32 and 1.68 printed uncertainties). The other checked properties
lie within their printed uncertainties. These differences are reported;
the benchmark does not adjust catalogue parameters to remove them.

NumPy and Matplotlib are required. Original panels and vector-tick registrations
are saved in `reference/`, so the benchmark runs without Python BurnMan, the
original PDF, or Poppler. To regenerate that independent source material,
place the supplied PDF at `~/deKoker_2013.pdf` and run:

```sh
python benchmarks/dekoker_2013/extract_reference.py
```

There is no continuous mixing fit, entropy recalibration or phase diagram in
this benchmark. Those calculations are the separate
[MgO–SiO₂ example](../../examples/example_dekoker_2013.py).

Source: [doi:10.1016/j.epsl.2012.11.026](https://doi.org/10.1016/j.epsl.2012.11.026).
