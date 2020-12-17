import os
import re

RETAIN_RAW = False
TARGETS = ["ncu-report.ksy"]
OUTPUTS = ['nsight_cuprof_report.py']

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("kaitai-struct-compiler --python-package \".\" --target python " + " ".join(TARGETS))

if not RETAIN_RAW:
    for filename in OUTPUTS:
        with open(filename, 'r') as file:
            text = file.read()
            text = text.replace('self._raw', '_raw')
        with open(filename, 'w') as file:
            file.write(text)
