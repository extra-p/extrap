# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import os
import re

RETAIN_RAW = False
TARGETS = ["ncu-report.ksy"]
OUTPUTS = ['nsight_cuprof_report.py']

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("kaitai-struct-compiler --python-package \".\" --target python " + " ".join(TARGETS))

for filename in OUTPUTS:
    with open(filename, 'r') as file:
        text = file.read()
        text = text.replace('self._raw', '_raw')
    with open(filename, 'w') as file:
        file.write(text)

if not RETAIN_RAW:
    for filename in OUTPUTS:
        with open(filename, 'r') as file:
            text = file.read()
            text = text.replace('self._raw', '_raw')
        with open(filename, 'w') as file:
            file.write(text)
