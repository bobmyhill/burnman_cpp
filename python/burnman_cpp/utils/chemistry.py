# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Chemical formula utilities ported from Python BurnMan, using native masses.

Reference: tests/reference/burnman.json.
"""

import numpy as np
from fractions import Fraction
from collections import Counter
from .._core.chemistry import (
    atomic_masses,
    IUPAC_element_order,
    dictionarize_formula,
    formula_mass,
)


def sum_formulae(formulae, amounts=None):
    """
    Adds together a set of formulae.

    :param formulae: List of chemical formulae.
    :type formulae: list of dictionary or counter objects
    :param amounts: List of amounts of each formula.
    :type amounts: list of floats

    :returns: The sum of the user-provided formulae
    :rtype: Counter object
    """
    if amounts is None:
        amounts = [1.0 for formula in formulae]
    else:
        assert len(formulae) == len(amounts)

    summed_formula = Counter()
    for i, formula in enumerate(formulae):
        summed_formula.update(
            Counter(
                {
                    element: amounts[i] * n_atoms
                    for (element, n_atoms) in formula.items()
                }
            )
        )
    return summed_formula


def convert_formula(formula, to_type="mass", normalize=False):
    """
    Converts a chemical formula from one type (mass or molar)
    into the other. Renormalises amounts if normalize=True.

    :param formula: A chemical formula.
    :type formula: dict or Counter object

    :param to_type: Conversion type, one of 'mass' or 'molar'.
    :type to_type: str

    :param normalize: Whether or not to normalize the converted formula to 1.
    :type normalize: bool

    :returns: The converted formula.
    :rtype: dict
    """

    if to_type == "mass":
        f = {
            element: n_atoms * atomic_masses[element]
            for (element, n_atoms) in formula.items()
        }
    elif to_type == "molar":
        f = {
            element: n_atoms / atomic_masses[element]
            for (element, n_atoms) in formula.items()
        }
    else:
        raise Exception(
            "Value of parameter to_type not recognised. "
            'Should be either "mass" or "molar".'
        )

    if normalize:
        s = np.sum([n for (element, n) in f.items()])
        f = {element: n / s for (element, n) in f.items()}

    return f


def site_occupancies_to_strings(
    site_species_names, site_multiplicities, endmember_occupancies
):
    """
    Converts a list of endmember site occupancies into a list
    of string representations of those occupancies.

    :param site_species_names: A list of list of strings,
        giving the names of the species which reside on each site.
        List of sites, each of which contains a list of the species
        occupying each site.
    :type site_species_names: 2D list of strings

    :param site_multiplicities: List of floats giving the multiplicity
        of each site. If 2D, must have the same shape as endmember_occupancies.
        If 1D, must be either the same length as the number of sites, or
        the same length as site_species_names
        (with an implied repetition of the same
        number for each species on a given site).
    :type site_multiplicities: 1D or 2D numpy array of floats

    :param endmember_occupancies: A list of site-species occupancies
        for each endmember. The first dimension loops over the endmembers, and the
        second dimension loops over the site-species occupancies for that endmember.
        The total number and order of occupancies must
        be the same as the strings in site_species_names.
    :type endmember_occupancies: 2D numpy array of floats

    :returns: A list of strings in standard burnman format.
        For example, [Mg]3[Al]2 would correspond to the
        classic two-site pyrope garnet.
    :rtype: list of strings
    """

    from sympy import nsimplify

    site_multiplicities = np.array(site_multiplicities)
    endmember_occupancies = np.array(endmember_occupancies)
    n_endmembers = endmember_occupancies.shape[0]

    if len(site_multiplicities.shape) == 1:
        # Site multiplicities should either be given on a per-site basis,
        # or a per-species basis
        if len(site_species_names) == len(site_multiplicities):
            site_mults = []

            for i, site in enumerate(site_species_names):
                for species in site:
                    site_mults.append(site_multiplicities[i])

            site_multiplicities = np.array(site_mults)

        elif len(endmember_occupancies[0]) != len(site_multiplicities):
            raise Exception(
                "Site multiplicities should either be given "
                "on a per-site basis or a per-species basis"
            )

        site_multiplicities = np.einsum(
            "i, j->ij", np.ones(n_endmembers), site_multiplicities
        )
    elif len(site_multiplicities.shape) == 2:
        if site_multiplicities.shape != endmember_occupancies.shape:
            raise Exception(
                "If site_multiplicities is 2D, it should have "
                "the same shape as endmember_occupancies. "
                "They currently have shapes "
                f"{site_multiplicities.shape} and "
                f"{endmember_occupancies.shape}."
            )
    else:
        raise Exception("Site multiplicities should either be 1D or 2D.")

    site_formulae = []
    for i_mbr, mbr_occupancies in enumerate(endmember_occupancies):
        i = 0
        site_formulae.append("")
        for site in site_species_names:
            amounts = mbr_occupancies[i : i + len(site)]
            mult = site_multiplicities[i_mbr, i]
            if np.abs(mult - 1.0) < 1.0e-12:
                mult = ""
            else:
                mult = str(nsimplify(mult))
            amounts /= sum(amounts)
            site_occupancy = formula_to_string(dict(zip(site, amounts)))
            site_formulae[-1] += "[{0}]{1}".format(site_occupancy, mult)
            i += len(site)

    return site_formulae


def compositional_array(formulae):
    """
    :param formulae: List of chemical formulae
    :type formulae: list of dicts

    :returns: Array of endmember formulae and a list of elements.
    :rtype: 2D numpy.array of floats and a list of strs
    """
    elements = []
    for formula in formulae:
        for element in formula:
            if element not in elements:
                elements.append(element)

    formula_array = ordered_compositional_array(formulae, elements)

    return formula_array, elements


def ordered_compositional_array(formulae, elements):
    """
    :param formulae: List of chemical formulae
    :type formulae: list of dicts

    :param elements : List of elements
    :type elements: list of strings

    :returns: Array of endmember formulae
    :rtype: 2D array of floats
    """
    formula_array = np.zeros(shape=(len(formulae), len(elements)))
    for idx, formula in enumerate(formulae):
        for element in formula:
            assert element in elements
            formula_array[idx][elements.index(element)] = formula[element]

    return formula_array


def formula_to_string(formula, use_fractions=True, significant_figures=2, latex=False):
    """
    :param formula: Chemical formula
    :type formula: dict or Counter

    :param use_fractions: Whether or not to use fractions in the output string.
        If False, numbers are rounded to the number of significant figures
        given by the significant_figures parameter.
    :type use_fractions: bool

    :param significant_figures: Number of significant figures to use if
        use_fractions=False.
    :type significant_figures: int

    :param latex: Whether or not to format the output string for use in LaTeX.
    :type latex: bool

    :returns: A formula string, with element order as given in the list
        IUPAC_element_order.
        If one or more keys in the dictionary are not one of the elements
        in the periodic table, then they are added at the end of the string.
    :rtype: str
    """

    if use_fractions:
        from sympy import nsimplify

    formula_string = ""
    if use_fractions:
        for e in IUPAC_element_order:
            if e in formula and np.abs(formula[e]) > 1.0e-12:
                if np.abs(formula[e] - 1.0) < 1.0e-12:
                    formula_string += e
                else:
                    val = str(nsimplify(formula[e]))
                    if latex:
                        val = "$_{{{" + val + "}}}$"
                    formula_string += e + val

        for e in formula:
            if e not in IUPAC_element_order:
                if e in formula and np.abs(formula[e]) > 1.0e-12:
                    if np.abs(formula[e] - 1.0) < 1.0e-12:
                        formula_string += e
                    else:
                        val = str(nsimplify(formula[e]))
                        if latex:
                            val = "$_{{{" + val + "}}}$"
                        formula_string += e + val

    else:
        for e in IUPAC_element_order:
            if e in formula and np.abs(formula[e]) > 1.0e-12:
                if np.abs(formula[e] - 1.0) < 1.0e-12:
                    formula_string += e
                else:
                    val = str(round(formula[e], significant_figures))
                    if latex:
                        val = "$_{{{" + val + "}}}$"
                    formula_string += e + val

        for e in formula:
            if e not in IUPAC_element_order:
                if e in formula and np.abs(formula[e]) > 1.0e-12:
                    if np.abs(formula[e] - 1.0) < 1.0e-12:
                        formula_string += e
                    else:
                        val = str(round(formula[e], significant_figures))
                        if latex:
                            val = "$_{{{" + val + "}}}$"
                        formula_string += e + val

    return formula_string


def sort_element_list_to_IUPAC_order(element_list):
    """
    :param element_list : List of elements.
    :type element_list: list

    :returns: List of elements sorted into IUPAC order
    :rtype: list
    """
    sorted_list = [e for e in IUPAC_element_order if e in element_list]
    assert len(sorted_list) == len(element_list)
    return sorted_list


def convert_fractions(composite, phase_fractions, input_type, output_type):
    """
    Takes a composite with a set of user defined molar, volume
    or mass fractions (which do not have to be the fractions
    currently associated with the composite) and
    converts the fractions to molar, mass or volume.

    Conversions to and from mass require a molar mass to be
    defined for all phases. Conversions to and from volume
    require set_state to have been called for the composite.

    :param composite: Composite for which fractions are to be defined.
    :type composite: :class:`~burnman.Composite`

    :param phase_fractions: List of input phase fractions
        (of type input_type).
    :type phase_fractions: list of floats

    :param input_type: Input fraction type. One of 'molar', 'mass' or 'volume'.
    :type input_type: str

    :param output_type: Output fraction type. One of 'molar', 'mass' or 'volume'.
    :type output_type: str

    :returns: List of output phase fractions (of type output_type)
    :rtype: list of floats
    """
    if input_type == "volume" or output_type == "volume":
        if not composite.has_state():
            raise Exception(
                composite.name
                + ".set_state(P, T) has not been called, so volume fractions are currently undefined. Exiting."
            )

    if input_type == "molar":
        molar_fractions = phase_fractions
    if input_type == "volume":
        total_moles = sum(
            volume_fraction / phase.molar_volume
            for volume_fraction, phase in zip(phase_fractions, composite.phases)
        )
        molar_fractions = [
            volume_fraction / (phase.molar_volume * total_moles)
            for volume_fraction, phase in zip(phase_fractions, composite.phases)
        ]
    if input_type == "mass":
        total_moles = sum(
            mass_fraction / phase.molar_mass
            for mass_fraction, phase in zip(phase_fractions, composite.phases)
        )
        molar_fractions = [
            mass_fraction / (phase.molar_mass * total_moles)
            for mass_fraction, phase in zip(phase_fractions, composite.phases)
        ]

    if output_type == "volume":
        total_volume = sum(
            molar_fraction * phase.molar_volume
            for molar_fraction, phase in zip(molar_fractions, composite.phases)
        )
        output_fractions = [
            molar_fraction * phase.molar_volume / total_volume
            for molar_fraction, phase in zip(molar_fractions, composite.phases)
        ]
    elif output_type == "mass":
        total_mass = sum(
            molar_fraction * phase.molar_mass
            for molar_fraction, phase in zip(molar_fractions, composite.phases)
        )
        output_fractions = [
            molar_fraction * phase.molar_mass / total_mass
            for molar_fraction, phase in zip(molar_fractions, composite.phases)
        ]
    elif output_type == "molar":
        output_fractions = molar_fractions

    return output_fractions
