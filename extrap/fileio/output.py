import re

import extrap.fileio.io_helper as io
from extrap.entities.experiment import Experiment
from extrap.util.exceptions import RecoverableError


class OutputFormatError(RecoverableError):
    NAME = 'Output Format Error'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def format_parameters(input_str: str, experiment: Experiment):
    param_list = [p.name for p in experiment.parameters]

    sep_options, format_options = _parse_options(input_str)
    param_sep = " " if sep_options is None else sep_options
    param_format = "" if format_options is None else format_options

    if param_format != "":
        param_list = [param_format.replace("{parameter}", p) for p in param_list]
    param_str = param_sep.join(param_list)

    return param_str


def format_points(options: str, experiment: Experiment):
    sep_options, format_options = _parse_options(options)
    points_sep = " | " if sep_options is None else sep_options
    points_format = "" if format_options is None else format_options

    sep_options, format_options = _parse_options(points_format)
    point_sep = ", " if sep_options is None else sep_options
    point_format = "" if format_options is None else format_options

    points = experiment.coordinates
    final_text = ""
    param_list = [p.name for p in experiment.parameters]

    for p in points:
        point = format_point(p, param_list, point_format)
        if points_format != "":
            placeholder = parse_outer_brackets(points_format)
            if placeholder:
                while placeholder[0][0] == "{" and placeholder[0][-1] == "}":
                    placeholder = parse_outer_brackets(placeholder[0])

            if len(placeholder) == 1 and re_point.search(*placeholder) is not None:
                final_text += points_format.replace(f"{{{placeholder[0]}}}", point_sep.join(point)) + points_sep
            else:
                final_text += points_format + points_sep
        else:
            final_text += "(" + point_sep.join(point) + ")" + points_sep
    final_text = final_text[:len(final_text) - len(points_sep)]

    return final_text


def format_point(p, param_list, point_format):
    if point_format != "":
        point = [point_format.format(parameter=param, coordinate=value)
                 for param, value in zip(param_list, p)]
    else:
        point = ["{:.2E}".format(value) for value in p]  # list contains a single point/coordinate
    return point


def format_measurements(input_str: str, experiment: Experiment, model):
    final_text = ""
    points = experiment.coordinates
    param_list = [p.name for p in experiment.parameters]

    sep_options, format_options = _parse_options(input_str)
    m_sep = "\n" if sep_options is None else sep_options
    m_format = "" if format_options is None else format_options

    sep_options, format_options = _parse_options(m_format)
    point_sep = " " if sep_options is None else sep_options
    point_format = "" if format_options is None else format_options

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
            print(m_format)
            m_string = m_format.format_map(SafeDict(mean=mean, median=median, std=std, min=min, max=max))  # TODO does not work
            # .replace("{mean}", "{:.2E}".format(mean)).replace("{median}", "{:.2E}".format(median)) \
            # .replace("{std}", "{:.2E}".format(std)).replace("{min}", "{:.2E}".format(min)) \
            # .replace("{max}", "{:.2E}".format(max))

            if point_string != "":
                m_string = m_string.replace("{%s}" % point_string, point_sep.join(point))

            final_text += m_string + m_sep

        else:
            final_text += "(" + point_sep.join(point) + ") " + \
                          f"Mean: {mean:.2E} Median: {median:.2E}" + m_sep
    final_text = final_text[:len(final_text) - len(m_sep)]

    return final_text


def parse_outer_brackets(input_str, remove_str=False):
    """return list of strings inside outermost pair(s) of curly brackets in input_str"""
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
    if x != -1:
        raise OutputFormatError("Unbalanced brackets!")
    for i in range(0, len(index), 2):
        result.append(input_str[index[i] + 1:index[i + 1]])

    if remove_str:  # remove string inside of curly brackets
        temp = list(input_str)
        for i in range(0, len(index), 2):
            temp[index[i] + 1: index[i + 1]] = ["" for j in range(index[i + 1] - index[i] - 1)]
        result = "".join(temp)

    return result


def parse_apostrophe(input_str):
    """ replace apostrophe outside of curly brackets """
    options = parse_outer_brackets(input_str)
    options = [f"{{{o}}}" for o in options]
    brackets = parse_outer_brackets(input_str, remove_str=True)
    brackets = brackets.replace("\\'", "\'")
    result = brackets.format(*options)

    return result


# single quotes are allowed in format or sep if they are escaped
re_format = re.compile(r"format:\s*'((?:{.*}|\\'|.*?)*)';?\s*")
re_sep = re.compile(r"sep:\s*'((?:\\'|.*?)*)';?\s*")

re_points = re.compile(r"(\?)?points(\s*:\s*(.*?))?")
re_point = re.compile(r"(\?)?point(\s*:\s*(.*?))?")
re_measurements = re.compile(r"measurements(\s*:\s*(.*?))?")
re_parameters = re.compile(r"(\?)?parameters(\s*:\s*(.*?))?")
re_options = re.compile(r"all|callpaths|metrics|parameters|functions")


def _parse_options(options):
    """Parses the attributes of the placeholders"""
    if options is None:
        return None, None

    format_result = re_format.search(options)
    if format_result:
        format_result = format_result.group(1)
        format_result = parse_apostrophe(format_result)
        options = re_format.sub('', options)

    sep_result = re_sep.search(options)
    if sep_result:
        sep_result = sep_result.group(1)
        sep_result = parse_apostrophe(sep_result)

    return sep_result, format_result


def fmt_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()
    print_str = parse_apostrophe(print_str)
    models = experiment.modelers[0].models

    # TODO how to implement brace escape?
    options = parse_outer_brackets(print_str)  # convert from input string to list
    text = ""

    print_coord = True
    print_param = True
    callpath_list = []
    metric_list = []

    # backward compatibility
    if re_options.fullmatch(print_str):
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

                elif re_points.fullmatch(o):
                    data = re_points.fullmatch(o)
                    coordinate_text = format_points(data.group(2), experiment)
                    temp, print_coord = _remove_duplicates_with_leading_question_mark(o, coordinate_text, print_coord,
                                                                                      temp)

                elif re_measurements.fullmatch(o):
                    data = re_measurements.fullmatch(o)
                    measurement_text = format_measurements(data.group(2), experiment, m)
                    temp = temp.replace(f"{{{o}}}", measurement_text)

                elif re_parameters.fullmatch(o):
                    data = re_parameters.fullmatch(o)
                    param_text = format_parameters(data.group(2), experiment)
                    temp, print_param = _remove_duplicates_with_leading_question_mark(o, param_text, print_param, temp)

                elif any(m in o for m in ("smape", "rrss", "rss", "ar2", "re")):
                    continue
                else:
                    raise OutputFormatError(f"Invalid placeholder: {o}")

            temp = temp.format_map(SafeDict(smape=m.hypothesis.SMAPE, rrss=m.hypothesis.rRSS, rss=m.hypothesis.RSS,
                               ar2=m.hypothesis.AR2, re=m.hypothesis.RE))
                                # TODO does not work

            text += (temp + "\n")

    text = text.replace("{{", "{").replace("}}", "}").replace("\\n", "\n").replace("\\t", "\t")

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
