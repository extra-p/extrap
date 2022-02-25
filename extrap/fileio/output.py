# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import re
from typing import List, Tuple, Union, Sequence

import extrap.fileio.io_helper as io
from extrap.entities.coordinate import Coordinate
from extrap.entities.experiment import Experiment
from extrap.util.exceptions import RecoverableError

"""
This module provides the custom output formatting for the experiments. 
You can find the explanation of the placeholders in docs/output-formatting.md
"""

class OutputFormatError(RecoverableError):
    NAME = 'Output Format Error'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def format_parameters(options: str, experiment: Experiment):
    param_list = [p.name for p in experiment.parameters]

    sep_options, format_options = _parse_options(options)
    param_sep = " " if sep_options is None else sep_options
    param_format = "" if format_options is None else format_options

    if param_format != "":
        param_list = [param_format.format(parameter=p) for p in param_list]
    param_str = param_sep.join(param_list)

    return param_str


def format_points(options: str, experiment: Experiment):
    sep_options, (format_options, format_str) = _parse_options(options, format_remove_str=True)
    points_sep = " | " if sep_options is None else sep_options

    points = experiment.coordinates
    final_text = ""
    param_list = [p.name for p in experiment.parameters]

    for p in points:
        if format_options:
            placeholder_replacement = []
            for o in format_options:
                if o.startswith("point"):
                    temp = format_point(o, p, param_list)
                else:
                    raise OutputFormatError(f"Invalid placeholder: {o}")
                placeholder_replacement.append(temp)
            final_text += format_str.format(*placeholder_replacement) + points_sep
        else:
            final_text += "(" + format_point("", p, param_list) + ")" + points_sep
    final_text = final_text[:len(final_text) - len(points_sep)]

    return final_text


def format_point(options: str, point_coordinate: Coordinate, param_list: Sequence[str]):
    sep_options, format_options = _parse_options(options)
    point_sep = ", " if sep_options is None else sep_options
    options = "" if format_options is None else format_options

    if options != "":
        point = [options.format(parameter=param, coordinate=value)
                 for param, value in zip(param_list, point_coordinate)]
    else:
        point = ["{:.2E}".format(value) for value in point_coordinate]  # list contains a single point/coordinate
    return point_sep.join(point)


def format_measurements(options: str, experiment: Experiment, model):
    final_text = ""
    points = experiment.coordinates
    param_list = [p.name for p in experiment.parameters]

    sep_options, (format_options, format_str) = _parse_options(options, format_remove_str=True)
    m_sep = "\n" if sep_options is None else sep_options

    for p in points:

        measurements = experiment.measurements[(model.callpath, model.metric)]
        measurement = next((me for me in measurements if me.coordinate == p), None)

        mean = 0 if measurement is None else measurement.mean
        median = 0 if measurement is None else measurement.median
        std = 0 if measurement is None else measurement.std
        min = 0 if measurement is None else measurement.minimum
        max = 0 if measurement is None else measurement.maximum

        if format_options:
            placeholder_replacement = []
            for o in format_options:
                if o.startswith("point"):
                    temp = format_point(o, p, param_list)
                elif any(k in o for k in ("mean", "median", "std", "min", "max")):
                    braced_o = f"{{{o}}}"
                    temp = braced_o.format(mean=mean, median=median, std=std, min=min, max=max)
                else:
                    raise OutputFormatError(f"Invalid placeholder: {o}")
                placeholder_replacement.append(temp)

            final_text += format_str.format(*placeholder_replacement) + m_sep
        else:
            final_text += f"({format_point('', p, param_list)}) Mean: {mean:.2E} Median: {median:.2E}{m_sep}"
    final_text = final_text[:len(final_text) - len(m_sep)]

    return final_text


def _parse_outer_brackets(input_str, remove_str=False) -> Union[List[str], Tuple[List[str], str]]:
    """return list of strings inside outermost pair(s) of curly brackets in input_str"""
    x = -1
    index = []
    result = []
    input_enumerator = enumerate(input_str)
    for i, char in input_enumerator:
        if char == '{':
            if len(input_str) > i + 1 and input_str[i + 1] == '{':
                # if two opening braces are encountered these are ignored
                next(input_enumerator)
            else:
                x += 1
                if x == 0:
                    index.append(i)
        elif char == '}':
            # look ahead to check if the closing brace is escaped
            temp_i = i
            is_escaped = False
            while len(input_str) > temp_i + 1 and input_str[temp_i + 1] == '}':
                temp_i, _ = next(input_enumerator)
                is_escaped = not is_escaped
            if not is_escaped:
                x -= 1
                if x == -1:
                    index.append(i)
    if x != -1:
        raise OutputFormatError("Unbalanced brackets!")
    for i in range(0, len(index), 2):
        result.append(input_str[index[i] + 1:index[i + 1]])

    if remove_str:  # remove string inside of curly brackets
        if not index:
            return [], input_str
        temp = []
        assert len(index) >= 2
        assert len(index) % 2 == 0
        temp.append(input_str[:index[0] + 1])
        for i in range(1, len(index) - 1, 2):
            temp.append(input_str[index[i]:index[i + 1] + 1])
        temp.append(input_str[index[-1]:])

        return result, "".join(temp)

    return result


def _parse_apostrophe(input_str, remove_str=False):
    """ replace apostrophe outside of curly brackets """
    options, brackets = _parse_outer_brackets(input_str, remove_str=True)
    brackets = brackets.replace("\\'", "\'")
    if remove_str:
        return options, brackets
    else:
        brackets = brackets.replace('{{', '{{{{').replace('}}', '}}}}')
        options = [f"{{{o}}}" for o in options]
        return brackets.format(*options)


# single quotes are allowed in format or sep if they are escaped
_re_format = re.compile(r"format:\s*'((?:{.*}|\\'|.*?)*)';?\s*")
_re_sep = re.compile(r"sep:\s*'((?:\\'|.*?)*)';?\s*")

_re_points = re.compile(r"(\?)?points(\s*:\s*(.*?))?")
_re_point = re.compile(r"(\?)?point(\s*:\s*(.*?))?")
_re_measurements = re.compile(r"measurements(\s*:\s*(.*?))?")
_re_parameters = re.compile(r"(\?)?parameters(\s*:\s*(.*?))?")
_re_legacy_options = re.compile(r"all|callpaths|metrics|parameters|functions")


def _parse_options(options, format_remove_str=False):
    """Parses the attributes of the placeholders"""
    if options is None:
        if format_remove_str:
            return None, (None, None)
        return None, None

    format_result = _re_format.search(options)
    if format_result:
        format_result = format_result.group(1)
        format_result = _parse_apostrophe(format_result, format_remove_str)
        options = _re_format.sub('', options)
    elif format_remove_str:
        format_result = (None, None)

    sep_result = _re_sep.search(options)
    if sep_result:
        sep_result = sep_result.group(1)
        sep_result = _parse_apostrophe(sep_result)

    return sep_result, format_result


def format_output(experiment: Experiment, printtype: str):
    print_str = printtype.lower()

    # backward compatibility
    if _re_legacy_options.fullmatch(print_str):
        return io.format_output(experiment, print_str.upper())

    models = experiment.modelers[0].models

    placeholder_options, print_str = _parse_outer_brackets(printtype,
                                                           remove_str=True)  # convert from input string to list
    text = ""

    print_coord = True
    print_param = True
    callpath_list = []
    metric_list = []

    for m in models.values():
        placeholder_results = []
        for o in placeholder_options:
            if o == "callpath":
                temp = m.callpath.name
            elif o == "?callpath":
                if not callpath_list or callpath_list[-1] != m.callpath.name:
                    temp = m.callpath.name
                else:
                    temp = " " * len(m.callpath.name)
                callpath_list.append(m.callpath.name)
            elif o == "metric":
                temp = m.metric.name
            elif o == "?metric":
                if not metric_list or metric_list[-1] != m.metric.name:
                    temp = m.metric.name
                else:
                    temp = " " * len(m.metric.name)
                metric_list.append(m.metric.name)
            elif o == "model":
                temp = m.hypothesis.function.to_string(*experiment.parameters)

            elif _re_points.fullmatch(o):
                data = _re_points.fullmatch(o)
                coordinate_text = format_points(data.group(2), experiment)
                temp, print_coord = _remove_duplicates_with_leading_question_mark(o, coordinate_text, print_coord)

            elif _re_measurements.fullmatch(o):
                data = _re_measurements.fullmatch(o)
                temp = format_measurements(data.group(2), experiment, m)

            elif _re_parameters.fullmatch(o):
                data = _re_parameters.fullmatch(o)
                param_text = format_parameters(data.group(2), experiment)
                temp, print_param = _remove_duplicates_with_leading_question_mark(o, param_text, print_param)

            elif any(k in o for k in ("smape", "rrss", "rss", "ar2", "re")):
                braced_o = f"{{{o}}}"
                temp = braced_o.format(smape=m.hypothesis.SMAPE, rrss=m.hypothesis.rRSS, rss=m.hypothesis.RSS,
                                       ar2=m.hypothesis.AR2, re=m.hypothesis.RE)
            else:
                raise OutputFormatError(f"Invalid placeholder: {o}")
            placeholder_results.append(temp)

        text += (print_str.format(*placeholder_results) + "\n")

    text = text.replace("\\n", "\n").replace("\\t", "\t")

    return text


def _remove_duplicates_with_leading_question_mark(o, text, is_print_enabled):
    if o[0] == '?':
        if is_print_enabled:
            temp = text
        else:
            temp = " " * len(text)
        is_print_enabled = False
    else:
        temp = text
    return temp, is_print_enabled
