import re

import extrap.fileio.io_helper as io
from extrap.entities.experiment import Experiment
from extrap.util.exceptions import RecoverableError


class OutputFormatError(RecoverableError):
    NAME = 'Output Format Error'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def format_parameters(input_str: str, experiment: Experiment):
    param_list = [p.name for p in experiment.parameters]
    param_format = ""
    param_sep = " "

    sep_options, format_options = _parse_options(input_str)
    if sep_options:
        param_sep = sep_options
    if format_options:
        param_format = format_options

    if param_format != "":
        param_list = [param_format.replace("{parameter}", p) for p in param_list]
    param_str = param_sep.join(param_list)

    return param_str


def format_points(options: str, experiment: Experiment):
    points_sep = " | "
    points_format = ""
    point_sep = ", "
    point_format = ""

    sep_options, format_options = _parse_options(options)
    if sep_options:
        points_sep = sep_options

    if format_options:
        points_format = format_options
        # TODO change the following to precompiled regex
        if re.search(r"sep\s*:\s*\'(.*?)\'", points_format):
            point_sep = re.search(r"sep\s*:\s*\'(.*?)\'", points_format).group(1)
        if re.search(r"format\s*:\s*\'(.*?)\'", points_format):
            point_format = re.search(r"format\s*:\s*\'(.*?)\'", points_format).group(1)

    points = experiment.coordinates
    final_text = ""
    param_list = [p.name for p in experiment.parameters]

    for p in points:
        point = format_point(p, param_list, point_format)
        if points_format != "":
            placeholder = parse_outer_brackets(points_format)
            final_text += points_format.replace(f"{{{placeholder[0]}}}", point_sep.join(point)) + points_sep
        else:
            final_text += "(" + point_sep.join(point) + ")" + points_sep
    final_text = final_text[:-1]

    return final_text


def format_point(p, param_list, point_format):
    if point_format != "":
        point = [point_format.format(parameter=param, coordinate=value)
                 for param, value in zip(param_list, p)]
    else:
        point = ["{:.2E}".format(value) for value in p]  # list contains a single point/coordinate
    return point


# TODO change the following method to use precompiled regex (see format_points)
def format_measurements(input_str: str, experiment: Experiment, model):
    final_text = "\n"
    points = experiment.coordinates
    param_list = [p.name for p in experiment.parameters]

    m_sep = "\n"
    point_sep = " "
    m_format = ""
    point_format = ""
    # TODO use _parse_options instead of the following
    # single quotes not allowed in format or sep
    sep_options, format_options = _parse_options(input_str)
    if sep_options:
        m_sep = sep_options

    if format_options:
        m_format = format_options
        if re.search(r"sep\s*:\s*\'(.*?)\'", m_format):
            point_sep = re.search(r"sep\s*:\s*\'(.*?)\'", m_format).group(1)
        if re.search(r"format\s*:\s*\'(.*?)\'", m_format):
            point_format = re.search(r"format\s*:\s*\'(.*?)\'", m_format).group(1)

    for p in points:
        point = format_point(p, param_list, point_format)

        measurements = experiment.measurements[(model.callpath, model.metric)]
        measurement = next((me for me in measurements if me.coordinate == p), None)

        mean = 0 if measurement is None else measurement.mean
        median = 0 if measurement is None else measurement.median
        std = 0 if measurement is None else measurement.std
        min = 0 if measurement is None else measurement.minimum
        max = 0 if measurement is None else measurement.maximum

        if m_format != "":
            point_string = next((str for str in parse_outer_brackets(m_format) if "point" in str), "")
            # TODO allow string formatting for below values
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


# single quotes are allowed in format or sep if they are escaped
re_format = re.compile(r"format:\s*'((?:{.*}|\\'|.*?)*)';?\s*")
re_sep = re.compile(r"sep:\s*'((?:\\'|.*?)*)';?\s*")


def _parse_options(options):
    """Parses the attributes of the placeholders"""
    if options is None:
        return None, None

    format_result = re_format.search(options)
    if format_result:
        format_result = format_result.group(1)
        options = re_format.sub('', options)

    sep_result = re_sep.search(options)
    if sep_result:
        sep_result = sep_result.group(1)

    return sep_result, format_result


def fmt_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()
    models = experiment.modelers[0].models

    options = parse_outer_brackets(print_str)  # convert from input string to list

    text = ""

    print_coord = True
    print_param = True
    callpath_list = []
    metric_list = []

    # TODO use precompiled regex
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
                    # TODO use precompiled regex
                elif re.fullmatch(r"(\?)?points(\s*:\s*(.*?))?", o):  # points
                    data = re.fullmatch(r"(\?)?points(?:\s*:\s*(.*?))?", o)
                    coordinate_text = format_points(data.group(2), experiment)
                    temp, print_coord = _remove_duplicates_with_leading_question_mark(o, coordinate_text, print_coord,
                                                                                      temp)

                elif re.fullmatch(r"measurements(\s*:\s*(.*?))?", o):  # measurements
                    data = re.fullmatch(r"measurements(\s*:\s*(.*?))?", o)
                    measurement_text = format_measurements(data.group(2), experiment, m)
                    temp = temp.replace(f"{{{o}}}", measurement_text)

                elif re.fullmatch(r"(\?)?parameters(\s*:\s*(.*?))?", o):  # parameters
                    data = re.fullmatch(r"(\?)?parameters(\s*:\s*(.*?))?", o)
                    param_text = format_parameters(data.group(2), experiment)
                    temp, print_param = _remove_duplicates_with_leading_question_mark(o, param_text, print_param, temp)

                elif any(o.startswith(m) for m in ("smape", "rrss", "rss", "ar2", "re")):
                    continue
                else:
                    raise OutputFormatError(f"Invalid placeholder: {o}")
            temp = temp.format(smape=m.hypothesis.SMAPE, rrss=m.hypothesis.rRSS, rss=m.hypothesis.RSS,
                               ar2=m.hypothesis.AR2, re=m.hypothesis.RE)
            text += (temp + "\n")

    return text


def _remove_duplicates_with_leading_question_mark(o, text, is_print_enabled, temp):
    braced_o = f"{{{o}}}"
    if o[0] == '?':
        if is_print_enabled:
            temp = temp.replace(braced_o, text)
        else:
            temp = temp.replace(braced_o, " " * len(text))
        is_print_enabled = False
    else:
        temp = temp.replace(braced_o, text)
    return temp, is_print_enabled
