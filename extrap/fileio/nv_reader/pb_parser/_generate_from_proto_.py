# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("protoc -I FileFormat --python_out=. --mypy_out=. FileFormat/*")
import_regex = re.compile(r"^(import\s+[\w_]+_pb2)", re.MULTILINE)
from_import_regex = re.compile(r"^from\s+([\w_]+_pb2)\s+import", re.MULTILINE)
from_typing_extensions_regex = re.compile(r"^from\s+typing_extensions\s+import", re.MULTILINE)
import_EnumTypeWrapper_regex = re.compile(
    r"(^from\s+google.protobuf.internal.enum_type_wrapper\s+import\s+\(\s*\n\s+)_EnumTypeWrapper", re.MULTILINE)
none_regex = re.compile(r"^None\s+=",re.MULTILINE)
for filename in os.listdir():
    if filename.endswith('.pyi'):
        with open(filename, 'r') as file:
            text = file.read()
        text = from_import_regex.sub(r"from .\1 import", text)
        text = from_typing_extensions_regex.sub("from typing import", text)
        text = import_EnumTypeWrapper_regex.sub(r"\1EnumTypeWrapper", text)
        with open(filename, 'w') as file:
            file.write(text)
    elif filename.endswith('.py'):
        with open(filename, 'r') as file:
            text = file.read()
        text = import_regex.sub(r"from . \1", text)
        text = none_regex.sub("None_ =",text)
        with open(filename, 'w') as file:
            file.write(text)
