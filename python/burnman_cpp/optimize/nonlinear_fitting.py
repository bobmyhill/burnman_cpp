# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Errors-in-variables nonlinear fitting using the native BurnMan LM loop."""

from abc import ABC, abstractmethod

from .._core.optimize.nonlinear_fitting import (
    abs_line_project,
    mle_estimate,
    find_mle,
    calculate_jacobian,
    nonlinear_least_squares_fit,
    confidence_prediction_bands,
    extreme_values,
)


class NonLinearModel(ABC):
    """Interface for a hypersurface model with observation covariances.

    Subclasses provide ``data``, ``data_covariances``, ``mle_tolerances`` and
    ``delta_params``. Optional ``flags`` select the model for each observation.
    Models with the same methods and attributes also work without subclassing.
    """

    @abstractmethod
    def get_params(self):
        """Return the current parameter vector."""

    @abstractmethod
    def set_params(self, param_values):
        """Set parameters, applying model bounds if needed."""

    @abstractmethod
    def function(self, x, flag=None):
        """Return a point on the model surface corresponding to x."""

    @abstractmethod
    def normal(self, x, flag=None):
        """Return the unit normal to the surface at x."""

    def validate(self):
        """Check that fitting attributes are present; native code checks values."""
        for name in ("data", "data_covariances", "mle_tolerances", "delta_params"):
            if not hasattr(self, name):
                raise AttributeError(f"Model requires {name}.")


__all__ = [
    "NonLinearModel",
    "abs_line_project",
    "mle_estimate",
    "find_mle",
    "calculate_jacobian",
    "nonlinear_least_squares_fit",
    "confidence_prediction_bands",
    "extreme_values",
    "plot_cov_ellipse",
    "corner_plot",
    "weighted_residual_plot",
    "plot_residuals",
]


def __getattr__(name):
    if name in {
        "plot_cov_ellipse",
        "corner_plot",
        "weighted_residual_plot",
        "plot_residuals",
    }:
        from . import _fitting_plots

        value = getattr(_fitting_plots, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
