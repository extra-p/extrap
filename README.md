# Extra-P

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/extra-p/extrap?style=plastic)
[![PyPI version](https://badge.fury.io/py/extrap.png)](https://badge.fury.io/py/extrap)
[![PyPI - License](https://img.shields.io/pypi/l/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub issues](https://img.shields.io/github/issues/extra-p/extrap?style=plastic)
![GitHub pull requests](https://img.shields.io/github/issues-pr/extra-p/extrap?style=plastic)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/extra-p/extrap/extrap?style=plastic)

Extra-P, automated empirical performance modeling for HPC and scientific
applications.

For questions regarding Extra-P please send a message to <extra-p-support@lists.parallel.informatik.tu-darmstadt.de>.

--------------------------------------------------------------------------------------------

### Table of Contents:

1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [How to use it](#Usage)
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


### Installation
Use the following command to install Extra-P and all required packages via `pip`.

```
python -m pip install extrap --upgrade
``` 

The `--upgrade` forces the installation of a new version if a previous version is already installed.

### Usage
Extra-P can be used in two ways, either using the command-line interface or the graphical user interface.
More information about the usage of Extra-P with both interfaces can be found in the [quick start guide](docs/quick-start.md).

#### Graphical User Interface
The graphical user interface can be started by executing the `extrap-gui` command.

#### Command Line Interface
The command line interface is available under the `extrap` command:

`extrap` _OPTIONS_ (`--cube` | `--text` | `--talpas` | `--json` | `--extra-p-3`) _FILEPATH_

You can use different input formats as shown in the examples below:
* Text files: `extrap --text test/data/text/one_parameter_1.txt`
* JSON files: `extrap --json test/data/json/input_1.JSON`
* Talpas files: `extrap --talpas test/data/talpas/talpas_1.txt`
* Create model and save it to text file at the given path: `extrap --out test.txt --text test/data/text/one_parameter_1.txt` 

The Extra-P command line interface has the following options.

| Positional arguments |                                              |
|----------------------|----------------------------------------------|
| _FILEPATH_           | Specify a file path for Extra-P to work with |

| Optional arguments                                        |                                              |
|-----------------------------------------------------------|----------------------------------------------|
| `-h`, `--help`                                            | Show help message and exit              |
| `--version`                                               | Show program's version number and exit       |
| `--log` {`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`} | Set program's log level (default: `WARNING`) |

| Input options                  |                                                                                |
|--------------------------------|--------------------------------------------------------------------------------|
| `--cube`                       | Load data from CUBE files                                                      |
| `--text`                       | Load data from text files                                                      |
| `--talpas`                     | Load data from Talpas data format                                              |
| `--json`                       | Load data from JSON or JSON Lines file                                         |
| `--extra-p-3`                  | Load data from Extra-P 3 experiment                                            |
| `--scaling` {`weak`, `strong`} | Set weak or strong scaling when loading data from CUBE files (default: `weak`) |

| Modeling options                                                     |                                                           |
|----------------------------------------------------------------------|-----------------------------------------------------------|
| `--median`                                                           | Use median values for computation instead of mean values  |
| `--modeler` {`Default`, `Basic`, `Refining`, `Multi-Parameter`}      | Selects the modeler for generating the performance models |
| `--options` _KEY_=_VALUE_ [_KEY_=_VALUE_ ...]                        | Options for the selected modeler                          |
| `--help-modeler` {`Default`, `Basic`, `Refining`, `Multi-Parameter`} | Show help for modeler options and exit                    |

| Output options                                                       |                                                                           |
|----------------------------------------------------------------------|---------------------------------------------------------------------------|
| `--out` _OUT_                                                        | Specify the output path for Extra-P results                               |
| `--print` {`all`, `callpaths`, `metrics`, `parameters`, `functions`} | Set which information should be displayed after modeling (default: `all`) |
 
                 
### License

[BSD 3-Clause "New" or "Revised" License](LICENSE)
