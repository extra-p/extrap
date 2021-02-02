license_header = """
# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
"""

import glob
import re

file_paths = glob.glob('../**/*.py', recursive=True)

license_header = license_header.strip()
license_header += '\n\n'

find_license_comment = re.compile(r"\s*(?:#.*?\n)+(?:#.*?Copyright.*?\n)(?:#.*?\n)+\s*")

for file_path in file_paths:
    if file_path.endswith('update_license_header.py') or file_path.endswith('_pb2.py'):
        continue
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    occurrences = find_license_comment.match(data)
    if occurrences:
        data_str = data[occurrences.end():]
    else:
        data_str = data

    if data_str.strip():
        data_str = license_header + data_str
    else:
        data_str = data_str.strip()
    # print(data_str)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data_str)
