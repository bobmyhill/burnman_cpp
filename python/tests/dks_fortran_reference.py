"""Apply the documented Fortran thermal law to the older pinned Python EOS.

The optional reference pin predates this correction. Preserve its full solid
implementation, replacing gamma/q and adding the analytic difference between
the Fortran and finite-strain thermal integrals. The actual updated Python
checkout is tested separately without this compatibility fixture.

Source: BurnMan revision 415b06103ea0127570475379406ef118167ac839,
stx_liquids/MgO-SiO2_param_files/aktintvto_220.f and pintvto_220.f.
"""

import numpy as np
from scipy.optimize import brentq


def install_fortran_reference(burnman, monkeypatch):
    module = burnman.eos.dks_solid
    probe = module._grueneisen_parameter_fast(1.0, 0.7, 1.4, 0.6)
    if np.isclose(probe, 1.4 * 0.7**0.6, rtol=1.0e-13, atol=0.0):
        return  # Already corrected; compare the actual Python implementation.

    original = module.DKS_S

    def integral_difference(volume, params):
        log_ratio = np.log(volume / params["V_0"])
        q = params["q_0"]
        power = params["grueneisen_0"] * (
            log_ratio if q == 0.0 else np.expm1(q * log_ratio) / q
        )
        finite = module._intgroverVdV(params["V_0"], volume, params["grueneisen_0"], q)
        return power - finite

    class FortranDKSSolid(original):
        def _grueneisen_parameter(self, pressure, temperature, volume, params):
            if not params.get("dks_s_power_law_grueneisen", True):
                return super()._grueneisen_parameter(
                    pressure, temperature, volume, params
                )
            return params["grueneisen_0"] * (volume / params["V_0"]) ** params["q_0"]

        def _volume_dependent_q(self, x, params):
            if not params.get("dks_s_power_law_grueneisen", True):
                return super()._volume_dependent_q(x, params)
            return params["q_0"]

        def volume(self, pressure, temperature, params):
            if not params.get("dks_s_power_law_grueneisen", True):
                return super().volume(pressure, temperature, params)
            residual = lambda v: self.pressure(temperature, v, params) - pressure
            bracket = burnman.utils.math.bracket(
                residual, params["V_0"], 0.01 * params["V_0"]
            )
            return brentq(residual, bracket[0], bracket[1], xtol=1.0e-20)

        def entropy(self, pressure, temperature, volume, params):
            value = super().entropy(pressure, temperature, volume, params)
            if params.get("dks_s_power_law_grueneisen", True):
                value += params["Cv"] * integral_difference(volume, params)
            return value

        def _helmholtz_energy(self, pressure, temperature, volume, params):
            value = super()._helmholtz_energy(pressure, temperature, volume, params)
            if params.get("dks_s_power_law_grueneisen", True):
                value -= (
                    params["Cv"]
                    * (temperature - params["T_0"])
                    * integral_difference(volume, params)
                )
            return value

        def shear_modulus(self, pressure, temperature, volume, params):
            if params.get("dks_s_power_law_grueneisen", True):
                return float("nan")
            return super().shear_modulus(pressure, temperature, volume, params)

    monkeypatch.setattr(module, "DKS_S", FortranDKSSolid)
    monkeypatch.setitem(burnman.eos.helper.eos_methods, "dks_s", FortranDKSSolid)
