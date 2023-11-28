import matplotlib.ticker as mticker


def frmt_scientific_coefficient(coefficient):
    """
    This method takes a coefficient and formats it into a string using scientific notation.
    """
    formater = mticker.ScalarFormatter(useMathText=True)
    formater.set_powerlimits((-3, 3))
    formatted_coefficients = "{}".format(formater.format_data(float(coefficient)))
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
