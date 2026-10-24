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

# Embed an HDF5 file for CMake and Make without requiring Python.
foreach(variable NAME INPUT OUTPUT)
  if(NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
    message(FATAL_ERROR "EmbedCovariance requires -D${variable}=...")
  endif()
endforeach()

file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "(..)" "0x\\1," bytes "${bytes}")
set(header [=[// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

]=])
get_filename_component(filename "${INPUT}" NAME)
string(APPEND header
  "// Generated from ${filename}; do not edit.\n#pragma once\nnamespace burnman::minerals::detail {\ninline constexpr unsigned char ${NAME}_covariance_data[] = {${bytes}};\n}\n")
get_filename_component(directory "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${directory}")
file(WRITE "${OUTPUT}" "${header}")
