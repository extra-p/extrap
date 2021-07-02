import re

import extrap.fileio.io_helper as io
from extrap.entities.experiment import Experiment


def format_coordinates(experiment: Experiment):
    coordinates = experiment.coordinates
    coordinate_text = ""

    for c in coordinates:
        coordinate = ["{:.2E}".format(value) for value in c]
        coordinate_text += "(" + ",".join(coordinate) + "),"
    coordinate_text = coordinate_text[:-1]

    return coordinate_text


def fmt_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()
    models = experiment.modelers[0].models
    options = re.findall(r"\{(.*?)\}", print_str)  # convert from input string to list
    text = ""

    print_coord = True
    print_param = True
    callpath_list = []
    metric_list = []

    # backward compatibility
    if re.fullmatch(r"all|callpaths|metrics|parameters|functions", print_str):
        text = io.format_output(experiment, print_str.upper())
    else:
        for m in models.values():
            temp = print_str

            for o in options:
                if o == "callpath":
                    temp = temp.replace("{callpath}", m.callpath.name)
                elif o == "?callpath":  # not sure
                    if not callpath_list or callpath_list[-1] != m.callpath.name:
                        temp = temp.replace("{?callpath}", m.callpath.name)
                    else:
                        temp = temp.replace("{?callpath}", " " * len(m.callpath.name))
                    callpath_list.append(m.callpath.name)
                elif o == "metric":
                    temp = temp.replace("{metric}", m.metric.name)
                elif o == "?metric":  # not sure
                    if not metric_list or metric_list[-1] != m.metric.name:
                        temp = temp.replace("{?metric}", m.metric.name)
                    else:
                        temp = temp.replace("{?metric}", " " * len(m.metric.name))
                    metric_list.append(m.metric.name)
                elif o == "model":
                    temp = temp.replace("{model}",
                                        m.hypothesis.function.to_string(*experiment.parameters))
                elif o == "points":
                    coordinate_text = format_coordinates(experiment)
                    temp = temp.replace("{points}", coordinate_text)
                elif o == "?points":  # not sure
                    coordinate_text = format_coordinates(experiment)
                    if print_coord:
                        temp = temp.replace("{?points}", coordinate_text)
                    else:
                        temp = temp.replace("{?points}", " " * len(coordinate_text))
                    print_coord = False
                elif o == "measurements":  # not sure
                    coordinates = experiment.coordinates
                    measurement_text = "\n"

                    for c in coordinates:
                        coordinate = ["{:.2E}".format(value) for value in c]
                        measurements = experiment.measurements[(m.callpath, m.metric)]
                        measurement = next((me for me in measurements if me.coordinate == c), None)
                        mean = 0 if measurement is None else measurement.mean
                        median = 0 if measurement is None else measurement.median
                        measurement_text += "Measurement point: (" + ",".join(coordinate) + ") " + \
                                            f"Mean: {mean:.2E} Median: {median:.2E} "

                    temp = temp.replace("{measurements}", measurement_text)
                elif o == "parameters":
                    param_string = " ".join([p.name for p in experiment.parameters])
                    temp = temp.replace("{parameters}", param_string)
                elif o == "?parameters":  # not sure
                    param_string = " ".join([p.name for p in experiment.parameters])
                    if print_param:
                        temp = temp.replace("{?parameters}", param_string)
                    else:
                        temp = temp.replace("{?parameters}", " " * len(param_string))
                    print_param = False
                elif o == "smape":
                    temp = temp.replace("{smape}", "{:.2E}".format(m.hypothesis.SMAPE))
                elif o == "rrss":
                    temp = temp.replace("{rrss}", "{:.2E}".format(m.hypothesis.rRSS))
                elif o == "rss":
                    temp = temp.replace("{rss}", "{:.2E}".format(m.hypothesis.RSS))
                elif o == "ar2":
                    temp = temp.replace("{ar2}", "{:.2E}".format(m.hypothesis.AR2))
                elif o == "re":
                    temp = temp.replace("{re}", "{:.2E}".format(m.hypothesis.RE))
                else:
                    raise ValueError("invalid input")

            text += (temp + "\n")

    return text
