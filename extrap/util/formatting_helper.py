# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2021-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

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
