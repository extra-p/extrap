import datetime

license_header = """
# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) {years}, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.
"""

import glob
import re
import subprocess

file_paths = glob.glob('../**/*.py', recursive=True)

license_header = license_header.strip()
license_header += '\n\n'

find_license_comment = re.compile(r"\s*(?:#.*?\n)+(?:#.*?Copyright.*?(\d\d\d\d).*?\n)(?:#.*?\n)+\s*")

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
        is_changed = subprocess.check_output(['git', 'status', '--porcelain', file_path])
        if is_changed:
            last_commit_year = str(datetime.datetime.now().year)
        else:
            last_commit = subprocess.check_output(
                ['git', 'log', '--follow', '--date=iso', '--pretty=format:"%cd"', '-1', file_path])
            last_commit_year = last_commit[1:5].decode()

        if not occurrences:
            first_commit = subprocess.check_output(
                ['git', 'log', '--follow', '--date=iso', '--pretty=format:"%cd"', '--diff-filter=A', file_path])
            first_commit_year = first_commit[1:5].decode()
            if not first_commit_year:
                first_commit_year = last_commit_year
        else:
            first_commit_year = occurrences[1]

        if first_commit_year == last_commit_year:
            if not last_commit_year:
                last_commit_year = datetime.datetime.now().year
            years = last_commit_year
        else:
            years = first_commit_year + '-' + last_commit_year
        data_str = license_header.format(years=years) + data_str
    else:
        data_str = data_str.strip()
    # print(data_str)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data_str)
