# Extra-P
**Automated performance modeling for HPC applications**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/extra-p/extrap?style=plastic)
[![PyPI version](https://badge.fury.io/py/extrap.png)](https://badge.fury.io/py/extrap)
[![PyPI - License](https://img.shields.io/pypi/l/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub issues](https://img.shields.io/github/issues/extra-p/extrap?style=plastic)
![GitHub pull requests](https://img.shields.io/github/issues-pr/extra-p/extrap?style=plastic)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/extra-p/extrap/Test%20extrap%20package?style=plastic)

[<img alt="Screenshot of Extra-P" src="https://github.com/extra-p/extrap/raw/master/docs/images/extra-p-2d.png" height="200" align="right" title="Screenshot of Extra-P"/>](docs/images/extra-p-2d.png)
Extra-P is an automatic performance-modeling tool that supports the user in the identification of *scalability bugs*. 
A scalability bug is a part of the program whose scaling behavior is unintentionally poor, 
that is, much worse than expected. A performance model is a formula that expresses a performance metric of interest 
such as execution time or energy consumption as a function of one or more execution parameters such as the size of the 
input problem or the number of processors. 

Extra-P uses measurements of various performance metrics at different execution configurations as input to generate 
performance models of code regions (including their calling context) as a function of the execution parameters. 
All it takes to search for scalability issues even in full-blown codes is to run a manageable number of small-scale 
performance experiments, launch Extra-P, and compare the asymptotic or extrapolated performance of the worst instances
to the expectations.

Extra-P generates not only a list of potential scalability bugs but also human-readable models for all 
performance metrics available such as floating-point operations or bytes sent by MPI calls that can be further 
analyzed and compared to identify the root causes of scalability issues.

Extra-P is developed by [TU Darmstadt](https://www.parallel.informatik.tu-darmstadt.de/) – 
in collaboration with [ETH Zurich](https://spcl.inf.ethz.ch/).

*For questions regarding Extra-P please send a message to <extra-p-support@lists.parallel.informatik.tu-darmstadt.de>.*

--------------------------------------------------------------------------------------------
### Table of Contents

1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [How to use it](#Usage)
4. [License](#License)

--------------------------------------------------------------------------------------------

### Requirements

* Python 3.7 or higher
* numpy
* pycubexr
* marshmallow
* tqdm
* PySide2 (for GUI)
* matplotlib (for GUI)
* pyobjc-framework-Cocoa (only for GUI on macOS)


### Installation
Use the following command to install Extra-P and all required packages via `pip`.

```
python -m pip install extrap --upgrade
``` 

The `--upgrade` forces the installation of a new version if a previous version is already installed.

### Usage
Extra-P can be used in two ways, either using the command-line interface or the graphical user interface.
More information about the usage of Extra-P with both interfaces can be found in the [quick start guide](docs/quick-start.md).

#### Graphical user interface
The graphical user interface can be started by executing the `extrap-gui` command.

#### Command line interface
The command line interface is available under the `extrap` command:

`extrap` _OPTIONS_ (`--cube` | `--text` | `--talpas` | `--json` | `--extra-p-3`) _FILEPATH_

You can use different input formats as shown in the examples below:
* Text files: `extrap --text test/data/text/one_parameter_1.txt`
* JSON files: `extrap --json test/data/json/input_1.JSON`
* Talpas files: `extrap --talpas test/data/talpas/talpas_1.txt`
* Create model and save it to text file at the given path: `extrap --out test.txt --text test/data/text/one_parameter_1.txt` 

The Extra-P command line interface has the following options.

| Arguments                                                            |                                              |
|----------------------------------------------------------------------|----------------------------------------------|
| **Positional**                                                       |                                              |
| _FILEPATH_                                                           | Specify a file path for Extra-P to work with |
| **Optional**                                                         |                                              |
| `-h`, `--help`                                                       | Show help message and exit                   |
| `--version`                                                          | Show program's version number and exit       |
| `--log` {`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`}            | Set program's log level (default: `WARNING`) |
| **Input options**                                                    |                                              |
| `--cube`                                                             | Load data from CUBE files                    |
| `--text`                                                             | Load data from text files                    |
| `--talpas`                                                           | Load data from Talpas data format            |
| `--json`                                                             | Load data from JSON or JSON Lines file       |
| `--extra-p-3`                                                        | Load data from Extra-P 3 experiment          |
| `--scaling` {`weak`, `strong`}                                       | Set weak or strong scaling when loading data from CUBE files (default: `weak`) |
| **Modeling options**                                                 |                                              |
| `--median`                                                           | Use median values for computation instead of mean values  |
| `--modeler` {`Default`, `Basic`, `Refining`, `Multi-Parameter`}      | Selects the modeler for generating the performance models |
| `--options` _KEY_=_VALUE_ [_KEY_=_VALUE_ ...]                        | Options for the selected modeler             |
| `--help-modeler` {`Default`, `Basic`, `Refining`, `Multi-Parameter`} | Show help for modeler options and exit       |
| **Output options**                                                   |                                              |
| `--out` _OUTPUT_PATH_                                                | Specify the output path for Extra-P results  |
| `--print` {`all`, `callpaths`, `metrics`, `parameters`, `functions`} | Set which information should be displayed after modeling (default: `all`) |
| `--save-experiment` _EXPERIMENT_PATH_                                | Saves the experiment including all models as Extra-P experiment (if no extension is specified, “.extra-p” is appended) | 
                 
### License

[BSD 3-Clause "New" or "Revised" License](LICENSE)
