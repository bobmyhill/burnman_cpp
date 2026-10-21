# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""The installed distribution and its compiled extension describe one release."""

from importlib.metadata import version

import burnman_cpp as bm


def test_native_and_distribution_versions_agree():
    assert bm.__version__ == version("burnman_cpp")
