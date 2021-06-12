from extrap.entities.experiment import Experiment
import extrap.fileio.io_helper as io
import re


def fmt_output(experiment:Experiment, printtype:str):
    pattern = r"\{(.*?)\}(\:\{(.*?)\})*"    # test!!!
    printtype = printtype.upper()

    if re.fullmatch(r"ALL|CALLPATHS|METRICS|PARAMETERS|FUNCTIONS", printtype):
        options = [printtype]
    elif re.fullmatch(pattern, printtype):
        pattern1 = r"\{(.*?)\}"     # test!!!
        options = re.findall(pattern1, printtype)
    else:
        raise ValueError('input invalid')

    text = ""

    # format_... ersetzen!!
    #
    # falls mehrere Elemente in options, und eines davon ist "ALL", werden
    # die anderen nicht mehr betrachtet?
    for opt in options:
        if opt == "ALL":
            text += io.format_all(experiment)
        elif opt == "CALLPATHS":
            text += (io.format_callpaths(experiment) + "\n" + "\t")
        elif opt == "METRICS":
            text += (io.format_metrics(experiment) + "\n" + "\t")
        elif opt == "PARAMETERS":
            text += (io.format_parameters(experiment) + "\n" + "\t")
        elif opt == "FUNCTIONS":
            text += (io.format_functions(experiment) + "\n" + "\t")
        else:
            raise ValueError('printtype does not exist')

    return text

