import re

import extrap.fileio.io_helper as io
from extrap.entities.experiment import Experiment


def format_parameters(input_str: str, experiment: Experiment):
    param_list = [p.name for p in experiment.parameters]
    param_format = ""
    param_sep = " "

    # single quotes not allowed in format or sep
    if re.search(r"format\s*:\s*\'(.*?)\'", input_str):
        param_format = re.search(r"format\s*:\s*\'(.*?)\'", input_str).group(1)
    if re.search(r"sep\s*:\s*\'(.*?)\'", input_str):
        param_sep = re.search(r"sep\s*:\s*\'(.*?)\'", input_str).group(1)

    if param_format != "":
        param_list = [param_format.replace("{parameter}", p) for p in param_list]
    param_str = param_sep.join(param_list)

    return param_str


def format_points(input_str: str, experiment: Experiment):
    points_sep = " | "
    points_format = ""
    point_sep = ", "
    point_format = ""

    # single quotes not allowed in format or sep
    sep_pattern = r"points\s*\:\s*sep\s*:\s*\'(.*?)\'"
    format_pattern = r"points\s*\:\s*(sep\s*:\s*\'.*?\'\;\s*)?format\s*\:\s*\'(.*)\'"

    if re.search(sep_pattern, input_str):
        points_sep = re.search(sep_pattern, input_str).group(1)

    if re.search(format_pattern, input_str):
        points_format = re.search(format_pattern, input_str).group(2)
        if re.search(r"sep\s*:\s*\'(.*?)\'", points_format):
            point_sep = re.search(r"sep\s*:\s*\'(.*?)\'", points_format).group(1)
        if re.search(r"format\s*:\s*\'(.*?)\'", points_format):
            point_format = re.search(r"format\s*:\s*\'(.*?)\'", points_format).group(1)

    points = experiment.coordinates
    final_text = ""
    param_list = [p.name for p in experiment.parameters]

    for p in points:
        point = ["{:.2E}".format(value) for value in p]  # list contains a single point/coordinate
        if point_format != "":
            point = [point_format.replace("{parameter}", param_list[i]).replace("{coordinate}", point[i])
                     for i in range(len(point))]
        if points_format != "":
            placeholder = parse_outer_brackets(points_format)
            final_text += points_format.replace("{%s}" % placeholder[0], point_sep.join(point)) + points_sep
        else:
            print(points_sep)
            final_text += "(" + point_sep.join(point) + ")" + points_sep
    final_text = final_text[:-1]

    return final_text


def format_measurements(input_str: str, experiment: Experiment, model):
    final_text = "\n"
    points = experiment.coordinates
    param_list = [p.name for p in experiment.parameters]

    m_sep = "\n"
    point_sep = " "
    m_format = ""
    point_format = ""

    # single quotes not allowed in format or sep
    sep_pattern = r"measurements\s*\:\s*sep\s*:\s*\'(.*?)\'"
    format_pattern = r"measurements\s*\:\s*(sep\s*:\s*\'.*?\'\;\s*)?format\s*\:\s*\'(.*)\'"

    if re.search(sep_pattern, input_str):
        m_sep = re.search(sep_pattern, input_str).group(1)

    if re.search(format_pattern, input_str):
        m_format = re.search(format_pattern, input_str).group(2)
        if re.search(r"sep\s*:\s*\'(.*?)\'", m_format):
            point_sep = re.search(r"sep\s*:\s*\'(.*?)\'", m_format).group(1)
        if re.search(r"format\s*:\s*\'(.*?)\'", m_format):
            point_format = re.search(r"format\s*:\s*\'(.*?)\'", m_format).group(1)

    for p in points:
        point = ["{:.2E}".format(value) for value in p]  # list contains a single point/coordinate
        if point_format != "":
            point = [point_format.replace("{parameter}", param_list[i]).replace("{coordinate}", point[i])
                     for i in range(len(point))]

        measurements = experiment.measurements[(model.callpath, model.metric)]
        measurement = next((me for me in measurements if me.coordinate == p), None)

        mean = 0 if measurement is None else measurement.mean
        median = 0 if measurement is None else measurement.median
        std = 0 if measurement is None else measurement.std
        min = 0 if measurement is None else measurement.minimum
        max = 0 if measurement is None else measurement.maximum

        if m_format != "":
            point_string = next((str for str in parse_outer_brackets(m_format) if "point" in str), "")
            final_text += m_format.replace("{%s}" % point_string, point_sep.join(point)) \
                              .replace("{mean}", "{:.2E}".format(mean)) \
                              .replace("{median}", "{:.2E}".format(median)) \
                              .replace("{std}", "{:.2E}".format(std)) \
                              .replace("{min}", "{:.2E}".format(min)) \
                              .replace("{max}", "{:.2E}".format(max)) + m_sep  # m_sep = \n doesnt work
        else:
            final_text += "Measurement point: (" + point_sep.join(point) + ") " + \
                          f"Mean: {mean:.2E} Median: {median:.2E}" + m_sep

    return final_text


def parse_outer_brackets(input_str):
    # return list of strings inside outermost pair(s) of curly brackets in input_str
    # assuming balanced brackets
    x = -1
    index = []
    result = []
    for i, char in enumerate(input_str):
        if char == '{':
            x += 1
            if x == 0:
                index.append(i)
        elif char == '}':
            x -= 1
            if x == -1:
                index.append(i)
    for i in range(0, len(index), 2):
        result.append(input_str[index[i] + 1:index[i + 1]])
    return result


def fmt_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()
    models = experiment.modelers[0].models

    options = parse_outer_brackets(print_str)  # convert from input string to list

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
                elif o == "?callpath":
                    if not callpath_list or callpath_list[-1] != m.callpath.name:
                        temp = temp.replace("{?callpath}", m.callpath.name)
                    else:
                        temp = temp.replace("{?callpath}", " " * len(m.callpath.name))
                    callpath_list.append(m.callpath.name)
                elif o == "metric":
                    temp = temp.replace("{metric}", m.metric.name)
                elif o == "?metric":
                    if not metric_list or metric_list[-1] != m.metric.name:
                        temp = temp.replace("{?metric}", m.metric.name)
                    else:
                        temp = temp.replace("{?metric}", " " * len(m.metric.name))
                    metric_list.append(m.metric.name)
                elif o == "model":
                    temp = temp.replace("{model}",
                                        m.hypothesis.function.to_string(*experiment.parameters))
                elif re.fullmatch(
                        r"(\?)?points\s*(:)?\s*(sep\s*:\s*\'(.*?)\')?\s*(;)?\s*(format\s*:\s*\'(.*?)\')?",
                        o):  # points
                    # remove duplicates with leading "?"

                    coordinate_text = format_points(o, experiment)
                    if o[0] == '?':
                        if print_coord:
                            temp = temp.replace("{%s}" % o, coordinate_text)
                        else:
                            temp = temp.replace("{%s}" % o, " " * len(coordinate_text))
                        print_coord = False
                    else:
                        temp = temp.replace("{%s}" % o, coordinate_text)

                elif re.fullmatch(
                        r"measurements\s*(:)?\s*(sep\s*:\s*\'(.*?)\')?\s*(;)?\s*(format\s*:\s*\'(.*?)\')?",
                        o):  # measurements

                    measurement_text = format_measurements(o, experiment, m)
                    temp = temp.replace("{%s}" % o, measurement_text)

                elif re.fullmatch(
                        r"(\?)?parameters\s*(:)?\s*(sep\s*:\s*\'(.*?)\')?\s*(;)?\s*(format\s*:\s*\'(.*?)\')?",
                        o):  # parameters
                    # remove duplicates with leading "?"

                    param_text = format_parameters(o, experiment)
                    if o[0] == '?':
                        if print_param:
                            temp = temp.replace("{%s}" % o, param_text)
                        else:
                            temp = temp.replace("{%s}" % o, " " * len(param_text))
                        print_param = False
                    else:
                        temp = temp.replace("{%s}" % o, param_text)

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
