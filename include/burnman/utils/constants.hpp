// ------------------------------------------------------
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

#ifndef BURNMAN_UTILS_CONSTANTS_HPP_INCLUDED
#define BURNMAN_UTILS_CONSTANTS_HPP_INCLUDED

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace burnman {

/**
 * @namespace burnman::constants
 * @brief Constants used in burnman
 */
namespace constants {

/**
 * @namespace burnman::constants::physics
 * @brief Physical constants
 *
 * @note Using CODATA 2022 values
 * @note To match the python implementation / scipy > 1.15 we use
 *       more digits for R and hbar than the CODATA2022 database.
 *       R is defined as k_B * N_A, and hbar as h / (2pi), so we
 *       can calculate them to full double precision.
 */
namespace physics {
/**
 * R in [J/mol/K]
 * CODATA2022: 8.314 462 618 ... J mol-1 K-1
 */
constexpr double gas_constant = 8.31446261815324;
/**
 * k_B in [J/K]
 */
constexpr double boltzmann = 1.380649e-23;
/**
 * N_A [1/mol]
 */
constexpr double avogadro = 6.02214076e23;
/**
 * Reduced planck constant (Dirac) in Js.
 * CODATA2022: 1.054 571 817 ...  x 10-34 J s
 */
constexpr double dirac = 1.0545718176461565e-34;
/**
 * Newtonian constant of gravitation, G in [m^3/kg/s^2].
 */
constexpr double gravitation = 6.67430e-11;
/**
 * Wavenumber conversion: 1 cm^-1 in [J/mol], at BurnMan's tabulated precision.
 */
constexpr double invcm = 11.9627;
} // namespace physics

/**
 * @namespace burnman::constants::precision
 * @brief Machine epsilon and tolerances used in burnman.
 */
namespace precision {
constexpr double double_eps = std::numeric_limits<double>::epsilon();
constexpr double rel_tolerance_eps = 4.0 * double_eps;
constexpr double abs_tolerance = 1.0e-12;
constexpr double inverseish_eps = 1.0e-5;
constexpr double logish_eps = 1.0e-7;
} // namespace precision

/**
 * @namespace burnman::constants::chemistry
 * @brief IUPAC element lists, etc.
 */
namespace chemistry {

// Atomic numbers for the Brosh high-pressure QSM constraints (NIST).
inline const std::unordered_map<std::string, int> atomic_numbers = {
    {"H", 1},    {"He", 2},   {"Li", 3},   {"Be", 4},   {"B", 5},
    {"C", 6},    {"N", 7},    {"O", 8},    {"F", 9},    {"Ne", 10},
    {"Na", 11},  {"Mg", 12},  {"Al", 13},  {"Si", 14},  {"P", 15},
    {"S", 16},   {"Cl", 17},  {"Ar", 18},  {"K", 19},   {"Ca", 20},
    {"Sc", 21},  {"Ti", 22},  {"V", 23},   {"Cr", 24},  {"Mn", 25},
    {"Fe", 26},  {"Co", 27},  {"Ni", 28},  {"Cu", 29},  {"Zn", 30},
    {"Ga", 31},  {"Ge", 32},  {"As", 33},  {"Se", 34},  {"Br", 35},
    {"Kr", 36},  {"Rb", 37},  {"Sr", 38},  {"Y", 39},   {"Zr", 40},
    {"Nb", 41},  {"Mo", 42},  {"Tc", 43},  {"Ru", 44},  {"Rh", 45},
    {"Pd", 46},  {"Ag", 47},  {"Cd", 48},  {"In", 49},  {"Sn", 50},
    {"Sb", 51},  {"Te", 52},  {"I", 53},   {"Xe", 54},  {"Cs", 55},
    {"Ba", 56},  {"La", 57},  {"Ce", 58},  {"Pr", 59},  {"Nd", 60},
    {"Pm", 61},  {"Sm", 62},  {"Eu", 63},  {"Gd", 64},  {"Tb", 65},
    {"Dy", 66},  {"Ho", 67},  {"Er", 68},  {"Tm", 69},  {"Yb", 70},
    {"Lu", 71},  {"Hf", 72},  {"Ta", 73},  {"W", 74},   {"Re", 75},
    {"Os", 76},  {"Ir", 77},  {"Pt", 78},  {"Au", 79},  {"Hg", 80},
    {"Tl", 81},  {"Pb", 82},  {"Bi", 83},  {"Po", 84},  {"At", 85},
    {"Rn", 86},  {"Fr", 87},  {"Ra", 88},  {"Ac", 89},  {"Th", 90},
    {"Pa", 91},  {"U", 92},   {"Np", 93},  {"Pu", 94},  {"Am", 95},
    {"Cm", 96},  {"Bk", 97},  {"Cf", 98},  {"Es", 99},  {"Fm", 100},
    {"Md", 101}, {"No", 102}, {"Lr", 103}, {"Rf", 104}, {"Db", 105},
    {"Sg", 106}, {"Bh", 107}, {"Hs", 108}, {"Mt", 109}, {"Ds", 110},
    {"Rg", 111}, {"Cn", 112}, {"Nh", 113}, {"Fl", 114}, {"Mc", 115},
    {"Lv", 116}, {"Ts", 117}, {"Og", 118},
};

// kg/mol; BurnMan's data/input_masses/atomic_masses.dat (NIST masses).
// Vc denotes a vacancy. Keep the reference table's precision for parity.
inline const std::unordered_map<std::string, double> atomic_masses = {
    {"Vc", 0.0},        {"H", 0.00100794}, {"He", 0.0040026}, {"Li", 0.006941},
    {"Be", 0.00901218}, {"B", 0.010811},   {"C", 0.0120107},  {"N", 0.0140067},
    {"O", 0.0159994},   {"F", 0.0189984},  {"Ne", 0.0201797}, {"Na", 0.0229898},
    {"Mg", 0.024305},   {"Al", 0.0269815}, {"Si", 0.0280855}, {"P", 0.0309738},
    {"S", 0.032065},    {"Cl", 0.035453},  {"Ar", 0.039948},  {"K", 0.0390983},
    {"Ca", 0.040078},   {"Sc", 0.0449559}, {"Ti", 0.047867},  {"V", 0.0509415},
    {"Cr", 0.0519961},  {"Mn", 0.054938},  {"Fe", 0.055845},  {"Co", 0.0589332},
    {"Ni", 0.0586934},  {"Cu", 0.063546},  {"Zn", 0.06538},   {"Ga", 0.069723},
    {"Ge", 0.07264},    {"As", 0.0749216}, {"Se", 0.07896},   {"Br", 0.079904},
    {"Kr", 0.083798},   {"Rb", 0.0854678}, {"Sr", 0.08762},   {"Y", 0.0889058},
    {"Zr", 0.091224},   {"Nb", 0.0929064}, {"Mo", 0.09596},   {"Ru", 0.10107},
    {"Rh", 0.102905},   {"Pd", 0.10642},   {"Ag", 0.107868},  {"Cd", 0.112411},
    {"In", 0.114818},   {"Sn", 0.11871},   {"Sb", 0.12176},   {"Te", 0.1276},
    {"I", 0.126904},    {"Xe", 0.131293},  {"Cs", 0.132905},  {"Ba", 0.137327},
    {"La", 0.138905},   {"Ce", 0.140116},  {"Pr", 0.140908},  {"Nd", 0.144242},
    {"Sm", 0.15036},    {"Eu", 0.151964},  {"Gd", 0.15725},   {"Tb", 0.158925},
    {"Dy", 0.1625},     {"Ho", 0.16493},   {"Er", 0.167259},  {"Tm", 0.168934},
    {"Yb", 0.173054},   {"Lu", 0.174967},  {"Hf", 0.17849},   {"Ta", 0.180948},
    {"W", 0.18384},     {"Re", 0.186207},  {"Os", 0.19023},   {"Ir", 0.192217},
    {"Pt", 0.195084},   {"Au", 0.196967},  {"Hg", 0.20059},   {"Tl", 0.204383},
    {"Pb", 0.2072},     {"Bi", 0.20898},   {"Th", 0.232038},  {"Pa", 0.231036},
    {"U", 0.238029}};

/**
 * IUPAC_element_order provides a list of all the elements.
 * Element order is based loosely on electronegativity,
 * following the scheme suggested by IUPAC, except that H
 * comes after the Group 16 elements, not before them.
 */
inline const std::vector<std::string> IUPAC_element_order = {
    "v",  "Og", "Rn", "Xe", "Kr", "Ar", "Ne", "He", // Group 18
    "Fr", "Cs", "Rb", "K",  "Na", "Li",             // Group 1 (not H)
    "Ra", "Ba", "Sr", "Ca", "Mg", "Be",             // Group 2
    "Lr", "No", "Md", "Fm", "Es", "Cf", "Bk", "Cm",
    "Am", "Pu", "Np", "U",  "Pa", "Th", "Ac", // Actinides
    "Lu", "Yb", "Tm", "Er", "Ho", "Dy", "Tb", "Gd",
    "Eu", "Sm", "Pm", "Nd", "Pr", "Ce", "La", // Lanthanides
    "Y",  "Sc",                               // Group 3
    "Rf", "Hf", "Zr", "Ti",                   // Group 4
    "Db", "Ta", "Nb", "V",                    // Group 5
    "Sg", "W",  "Mo", "Cr",                   // Group 6
    "Bh", "Re", "Tc", "Mn",                   // Group 7
    "Hs", "Os", "Ru", "Fe",                   // Group 8
    "Mt", "Ir", "Rh", "Co",                   // Group 9
    "Ds", "Pt", "Pd", "Ni",                   // Group 10
    "Rg", "Au", "Ag", "Cu",                   // Group 11
    "Cn", "Hg", "Cd", "Zn",                   // Group 12
    "Nh", "Tl", "In", "Ga", "Al", "B",        // Group 13
    "Fl", "Pb", "Sn", "Ge", "Si", "C",        // Group 14
    "Mc", "Bi", "Sb", "As", "P",  "N",        // Group 15
    "Lv", "Po", "Te", "Se", "S",  "O",        // Group 16
    "H",                                      // Hydrogen
    "Ts", "At", "I",  "Br", "Cl", "F"         // Group 17
};

} // namespace chemistry

} // namespace constants
} // namespace burnman

#endif // BURNMAN_UTILS_CONSTANTS_HPP_INCLUDED
