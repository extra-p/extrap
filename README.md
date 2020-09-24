# Extra-P

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/extra-p/extrap?style=plastic)
[![PyPI version](https://badge.fury.io/py/extrap.png)](https://badge.fury.io/py/extrap)
[![PyPI - License](https://img.shields.io/pypi/l/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub issues](https://img.shields.io/github/issues/extra-p/extrap?style=plastic)
![GitHub pull requests](https://img.shields.io/github/issues-pr/extra-p/extrap?style=plastic)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/extra-p/extrap/extrap?style=plastic)

Put project description here.

--------------------------------------------------------------------------------------------

### Table of Contents:

1. [Requirements](#Requirements)
2. [How to use it](#Usage)
3. [Notes](#Notes)
4. [License](#License)

--------------------------------------------------------------------------------------------

### Requirements

* Python 3.7
* PySide2 (only if using GUI)
* maplotlib
* numpy
* pycubexr
* marshmallow
* tqdm

### Usage

* `extrap --text C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt` Text files.
* `extrap --json C:\Users\Admin\git\Extra-P\data\json\input_1.JSON` JSON files.
* `extrap --talpas C:\Users\Admin\git\Extra-P\data\talpas\talpas_1.txt` Talpas files.
* `extrap --out C:\Users\Admin\Desktop\test.txt --text C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt` Create model and save it to text file at the given path.


extrap.py OPTIONS (--cube | --text | --talpas | --json | --extra-p-3) FILEPATH
        
OPTIONS:

*  `-h, --help`            show help message and exit
*  `--log LOG_LEVEL`       set program's log level [INFO (default), DEBUG]
*  `--version`             show program's version number and exit
*  `--help-options {Basic,Refining,Multi-Parameter,Default}`
                        shows help for modeler options
*  `--cube`                load data from cube files
*  `--text`                load data from text files
*  `--talpas`              load data from Talpas data format
*  `--json`                load data from JSON or JSON Lines file
*  `--extra-p-3`           load data from Extra-P 3 experiment
*  `--modeler {Basic,Refining,Multi-Parameter,Default}` selects a modeler
*  `--options KEY=VALUE [KEY=VALUE ...]` sets options for the selected modeler
*  `--scaling {weak,strong}`
                        set weak or strong scaling when loading data from cube
                        files [weak (default), strong]
*  `--median`              use median values for computation instead of mean
                        values
*  `--out OUT`             specify the output path for Extra-P results
*  `--print {all,callpaths,metrics,parameters,functions}`
                        set which information should be displayed after
                        modeling [all (default), callpaths, metrics,
                        parameters, functions]

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
* `extrap --text C:\Users\Admin\git\Extra-P\data\text\two_parameter_1.txt`
* `extrap --cube C:\Users\Admin\git\Extra-P\data\cube\kripke\`
* `extrap --cube C:\Users\Admin\git\Extra-P\data\cube\blast\`


### License

[BSD 3-Clause "New" or "Revised" License](LICENSE)
