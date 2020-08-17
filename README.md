### Extra-P

![Python package](https://github.com/MeaParvitas/Extra-P/workflows/Python%20package/badge.svg?branch=master)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/extrap?style=plastic)](https://badge.fury.io/py/extrap)
[![PyPI version](https://badge.fury.io/py/extrap.png)](https://badge.fury.io/py/extrap)
[![PyPI - License](https://img.shields.io/pypi/l/extrap?style=plastic)](https://badge.fury.io/py/extrap)

Put project description here.

--------------------------------------------------------------------------------------------

### Table of Contents:

1. [Requirements](#Requirements)
2. [How to use it](#Usage)
3. [Notes](#Notes)
4. [License](#License)

--------------------------------------------------------------------------------------------

### Requirements

* Python 3
* PyQt 5 (not anymore! change this)
* maplotlib
* numpy

### Usage

* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt` Text files.
* `extrap --json --path C:\Users\Admin\git\Extra-P\data\json\input_1.JSON` JSON files.
* `extrap --talpas --path C:\Users\Admin\git\Extra-P\data\talpas\talpas_1.txt` Talpas files.
* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt --out C:\Users\Admin\Desktop\test.txt` Create model and save it to text file at the given path.

Extra-P

optional arguments:

* `-h, --help` show this help message and exit
* `--log LOG_LEVEL` set program's log level [INFO (default), DEBUG]
* `--version` show program's version number and exit
* `--path PATH` specify a file path for Extra-P to work with
* `--scaling SCALING` set weak or strong scaling when loading data from cube files [WEAK (default), STRONG]
* `--cube` load data from cube files
* `--median` use median values for computation instead of mean values
* `--text` load data from text files
* `--talpas` load data from talpas data format
* `--json` load data from json file
* `--out OUT` specify the output path for Extra-P results
* `--print PRINT_TYPE` set which information should be displayed after modeling [ALL (default), CALLPATHS, METRICS, PARAMETERS, FUNCTIONS]

### Build Extra-P package

1. `python setup.py sdist bdist_wheel` Create package from code.
2. `python -m twine upload --repository testpypi dist/*` Upload package to python index. Need to specify username, password and do not forget to update the version of the package.

#### Build virtual env to test package

This command only works in windows shell...

1. `python -m venv /tmppython` Create a new virtual python environment to test the code.
2. `\tmppython\Scripts\activate` Activate the virtual environment to use it for testing.
3. `deactivate` Deactivate the virtual environment.

#### Install the Extra-P package

1. `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps extrap-meaparvitas --upgrade` Install the Extra-P package. The ``--upgrade` forces the installation of a new version if a previous version is already installed.
2. `extrap` To run the command line version of Extra-P.

#### Installing from a local src tree

1. `pip install -e <path>` Install package from a local src tree via a sim link.

#### Command Line stuff:

* PyQT needs to be install in order for the gui to display when running the extrapgui command in a terminal.
* python -m pip install --user --upgrade twine
* python -m pip install --user --upgrade setuptools wheel
* `pip install -e C:\Users\Admin\git\extrap\`
* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\two_parameter_1.txt`
* `extrap --cube --path C:\Users\Admin\git\Extra-P\data\cube\kripke\`
* `extrap --cube --path C:\Users\Admin\git\Extra-P\data\cube\blast\`

### License

[BSD 3-Clause "New" or "Revised" License](LICENSE)