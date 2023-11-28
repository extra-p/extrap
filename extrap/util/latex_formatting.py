# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import matplotlib.ticker as mticker


def frmt_scientific_coefficient(coefficient):
    """
    This method takes a coefficient and formats it into a string using scientific notation.
    """
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    formatted_coefficients = "{}".format(
        formatter.format_data(float(coefficient)))
    coreff_terms = formatted_coefficients.split(" ")
    new_coeff = ""
    if not coreff_terms[0][:1].isnumeric():
        coeff = coreff_terms[0][1:]
        try:
            coeff = "{:.3f}".format(float(coeff))
        except ValueError:
            pass
        new_coeff += "-"
        new_coeff += coeff
        for i in range(len(coreff_terms)):
            if i != 0:
                new_coeff += coreff_terms[i]
        return new_coeff
    else:
        coeff = coreff_terms[0]
        try:
            coeff = "{:.3f}".format(float(coeff))
        except ValueError:
            pass
        new_coeff += coeff
        for i in range(len(coreff_terms)):
            if i != 0:
                new_coeff += coreff_terms[i]
        return new_coeff
