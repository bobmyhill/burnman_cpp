# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Pressure standards with native inversion and measurement covariance.

Pressure is in Pa, volume in m3/mol and temperature in K. Input covariance
orders are [V,T] for pressure() and [P,T] for volume(); returned 3x3 matrices
use [P,V,T] and [V,P,T], respectively, as in Python BurnMan.
"""

from .._core import Calibrant as _NativeCalibrant


class Calibrant:
    """A pressure or volume function, its parameter dictionary and inverse.

    ``calibrant_function(value, temperature, params)`` returns the quantity
    selected by ``calibrant_function_return_type`` ('pressure' or 'volume').
    Custom Python functions and mutable parameter dictionaries are supported;
    inversion and covariance propagation execute in the native library.
    """

    def __init__(self, calibrant_function, calibrant_function_return_type, params):
        self.calibrant_function = calibrant_function
        self.params = params
        self._return_type = calibrant_function_return_type
        self._catalogue_model = False
        self._as_native()  # validate the function return type at construction
        if self._return_type == "pressure":
            self.pressure_function = calibrant_function
            self.volume_function = self._volume_using_pressure_function
        else:
            self.volume_function = calibrant_function
            self.pressure_function = self._pressure_using_volume_function

    @staticmethod
    def _native_function(function, return_type, params):
        # Preserve arbitrary Python values in custom parameter dictionaries.
        # The native inversion only needs V_0; the callback receives the actual
        # supplied mapping, including arrays or other function-specific values.
        native_params = {"V_0": params["V_0"]} if "V_0" in params else {}
        return _NativeCalibrant(
            lambda value, temperature, unused: function(value, temperature, params),
            return_type,
            native_params,
        )

    def _set_native(self, native):
        self._native = native
        self._return_type = native.calibrant_function_return_type
        self._catalogue_model = True
        self.params = native.params
        self.pressure_function = native.pressure_function
        self.volume_function = native.volume_function
        self.calibrant_function = (
            self.pressure_function
            if self._return_type == "pressure"
            else self.volume_function
        )

    def _as_native(self):
        if self._catalogue_model:
            self._native.params = self.params
        else:
            self._native = self._native_function(
                self.calibrant_function, self._return_type, self.params
            )
        return self._native

    def _volume_using_pressure_function(self, pressure, temperature, params):
        return self._native_function(self.pressure_function, "pressure", params).volume(
            pressure, temperature
        )

    def _pressure_using_volume_function(self, volume, temperature, params):
        return self._native_function(self.volume_function, "volume", params).pressure(
            volume, temperature
        )

    def pressure(self, volume, temperature, VT_covariance=None):
        """Return pressure, optionally with covariance in [P,V,T] order."""
        return self._as_native().pressure(volume, temperature, VT_covariance)

    def volume(self, pressure, temperature, PT_covariance=None):
        """Return volume, optionally with covariance in [V,P,T] order."""
        return self._as_native().volume(pressure, temperature, PT_covariance)
