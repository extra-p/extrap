# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import math
import re


def replace_braces_recursively(name, lb, rb, replacement="{lb}…{rb}"):
    depth = 0
    start = -1
    replacement = replacement.format(lb=lb, rb=rb)
    i = 0
    while i < len(name):
        elem = name[i]
        if elem == lb:
            depth += 1
            if start == -1:
                start = i
        elif elem == rb:
            depth -= 1
            if start != -1 and depth == 0:
                if start + 1 != i:
                    name = name[0:start:] + replacement + name[i + 1:len(name):]
                    i = start + len(replacement) - 1
                start = -1
        i += 1
    return name


def replace_method_parameters(name, replacement="{lb}…{rb}"):
    name = re.sub(r"\[with.*\]", replacement.format(lb='[with', rb=']'), name)
    name = replace_braces_recursively(name, '<', '>', replacement)
    name = replace_braces_recursively(name, '(', ')', replacement)
    return name


def format_number_plain_text(number, precision=3, scientific_notation=4, no_integer=False):
    if not math.isfinite(number):
        return str(number)
    elif number == 0 or 10 ** -precision < abs(number) < 10 ** scientific_notation:
        if not no_integer:
            if isinstance(number, int):
                return str(number)
            number = float(number)
            if number.is_integer():
                return str(int(number))
        return '{:.{}f}'.format(number, precision)
    else:
        mantissa, exponent = '{:.{}e}'.format(number, precision).split("e")
        exponent_sign = exponent[0]
        exponent = exponent[1:].lstrip('0')
        if exponent == '':
            return mantissa
        else:
            if exponent_sign != '+':
                exponent = exponent_sign + exponent
            return mantissa + "×10" + make_exponent(exponent)


def format_number_ascii(number, precision=3, scientific_notation=4, no_integer=False):
    if not math.isfinite(number):
        return str(number)
    elif number == 0 or 10 ** -precision < abs(number) < 10 ** scientific_notation:
        if not no_integer:
            if isinstance(number, int):
                return str(number)
            number = float(number)
            if number.is_integer():
                return str(int(number))
        return '{:.{}f}'.format(number, precision)
    else:
        return '{:.{}e}'.format(number, precision)


def make_exponent(exponent):
    exponent = exponent.replace('0', '⁰')
    exponent = exponent.replace('1', '¹')
    exponent = exponent.replace('2', '²')
    exponent = exponent.replace('3', '³')
    exponent = exponent.replace('4', '⁴')
    exponent = exponent.replace('5', '⁵')
    exponent = exponent.replace('6', '⁶')
    exponent = exponent.replace('7', '⁷')
    exponent = exponent.replace('8', '⁸')
    exponent = exponent.replace('9', '⁹')
    exponent = exponent.replace('-', '⁻')
    exponent = exponent.replace('.', ' ̇ ')
    exponent = exponent.replace('/', '⸍')
    return exponent


def format_number_html(number, precision=3, scientific_notation=4, no_integer=False):
    if not math.isfinite(number):
        return str(number)
    elif 10 ** -precision < abs(number) < 10 ** scientific_notation:
        if not no_integer:
            if isinstance(number, int):
                return str(number)
            number = float(number)
            if number.is_integer():
                return str(int(number))
        return '{:.{}f}'.format(number, precision)
    else:
        mantissa, exponent = '{:.{}e}'.format(number, precision).split("e")
        exponent_sign = exponent[0]
        exponent = exponent[1:].lstrip('0')
        if exponent == '':
            return mantissa
        else:
            if exponent_sign != '+':
                exponent = exponent_sign + exponent
            return f"{mantissa}<small>&times;10</small><sup>{exponent}</sup>"
