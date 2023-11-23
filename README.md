# Extra-P

**Automated performance modeling for HPC applications**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/extra-p/extrap?style=plastic)
[![PyPI version](https://badge.fury.io/py/extrap.png)](https://badge.fury.io/py/extrap)
[![PyPI - License](https://img.shields.io/pypi/l/extrap?style=plastic)](https://badge.fury.io/py/extrap)
![GitHub issues](https://img.shields.io/github/issues/extra-p/extrap?style=plastic)
![GitHub pull requests](https://img.shields.io/github/issues-pr/extra-p/extrap?style=plastic)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/extra-p/extrap/python-package.yml?style=plastic)

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

The following video on the Laboratory for Parallel Programming @ TUDa [YouTube](
https://www.youtube.com/@parallel_tuda) channel provides a quick introduction to Extra-P.

<a href="https://www.youtube.com/watch?v=Cv2YRCMWqBM"><img src="https://img.youtube.com/vi/Cv2YRCMWqBM/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Extra-P is developed by [TU Darmstadt](https://www.parallel.informatik.tu-darmstadt.de/) – 
in collaboration with [ETH Zurich](https://spcl.inf.ethz.ch/).

*For questions regarding Extra-P, please send a message to <extra-p-support@lists.parallel.informatik.tu-darmstadt.de>.*

--------------------------------------------------------------------------------------------

### Table of Contents

1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [How to use it](#Usage)
4. [License](#License)
5. [Citation](#Citation)
6. [Publications](#Publications)

--------------------------------------------------------------------------------------------

### Requirements

* Python 3.8 or higher
* numpy
* pycubexr
* marshmallow
* packaging
* tqdm
* PySide6 (for GUI)
* matplotlib (for GUI)
* pyobjc-framework-Cocoa (only for GUI on macOS)

### Installation

Use the following command to install Extra-P and all required packages via `pip`.

```
python -m pip install extrap --upgrade
``` 

The `--upgrade` forces the installation of a new version if a previous version is already installed.

### Usage

Extra-P can be used in two ways, either using the command-line interface or the graphical user interface. More
information about the usage of Extra-P with both interfaces can be found in the [quick start guide](docs/quick-start.md).

> **Note**  
> Extra-P is designed for weak-scaling, therefore, directly modeling of strong-scaling behaviour is not supported.
> Instead of modeling the runtime of your strong-scaling experiment, you can model the resource consumption, i.e.,
> the runtime *times* the number of processors.


#### Graphical user interface

The graphical user interface can be started by executing the `extrap-gui` command.

#### Command line interface

The command line interface is available under the `extrap` command:

`extrap` _OPTIONS_ (`--cube` | `--text` | `--talpas` | `--json` | `--extra-p-3`) _FILEPATH_

You can use different input formats as shown in the examples below:

* Text files: `extrap --text test/data/text/one_parameter_1.txt`
* JSON files: `extrap --json test/data/json/input_1.JSON`
* Talpas files: `extrap --talpas test/data/talpas/talpas_1.txt`
* Create model and save it to text file at the given
  path: `extrap --out test.txt --text test/data/text/one_parameter_1.txt`

The Extra-P command line interface has the following options.

| Arguments                                                            |                                              |
|----------------------------------------------------------------------|----------------------------------------------|
| **Positional**                                                       |                                              
| _FILEPATH_                                                           | Specify a file path for Extra-P to work with 
| **Optional**                                                         |                                              
| `-h`, `--help`                                                       | Show help message and exit                   
| `--version`                                                          | Show program's version number and exit       
| `--log` {`debug`, `info`, `warning`, `error`, `critical`}            | Set program's log level (default: `warning`) 
| **Input options**                                                    |                                              
| `--cube`                                                             | Load a set of CUBE files and generate a new experiment
| `--extra-p-3`                                                        | Load data from Extra-P 3 (legacy) experiment
| `--json`                                                             | Load data from JSON or JSON Lines input file
| `--talpas`                                                           | Load data from Talpas data format
| `--text`                                                             | Load data from text input file
| `--experiment`                                                       | Load Extra-P experiment and generate new models
| `--scaling` {`weak`, `strong`}                                       | Set weak or strong scaling when loading data from CUBE files (default: `weak`) 
| **Modeling options**                                                 |                                              
| `--median`                                                           | Use median values for computation instead of mean values  
| `--modeler` {`default`, `basic`, `refining`, `multi-parameter`}      | Selects the modeler for generating the performance models 
| `--options` _KEY_=_VALUE_ [_KEY_=_VALUE_ ...]                        | Options for the selected modeler             
| `--help-modeler` {`default`, `basic`, `refining`, `multi-parameter`} | Show help for modeler options and exit       
| **Output options**                                                   |                                              
| `--out` _OUTPUT_PATH_                                                | Specify the output path for Extra-P results  
| `--print` {`all`, `callpaths`, `metrics`, `parameters`, `functions`, _FORMAT_STRING_} | Set which information should be displayed after modeling. Use one of the predefined values or specify a formatting string using placeholders (see [docs/output-formatting.md](docs/output-formatting.md)) (default: `all`).
| `--save-experiment` <i>EXPERIMENT_PATH</i>                           | Saves the experiment including all models as Extra-P experiment (if no extension is specified, “.extra-p” is appended) 
| `--model-set-name` _NAME_                                            | Set the name of the generated set of models when outputting an experiment (default: “New model”)

### License

[BSD 3-Clause "New" or "Revised" License](LICENSE)

### Citation

Please cite Extra-P in your publications if it helps your research:

    @inproceedings{calotoiu_ea:2013:modeling,
      author = {Calotoiu, Alexandru and Hoefler, Torsten and Poke, Marius and Wolf, Felix},
      month = {November},
      title = {Using Automated Performance Modeling to Find Scalability Bugs in Complex Codes},
      booktitle = {Proc. of the ACM/IEEE Conference on Supercomputing (SC13), Denver, CO, USA},
      year = {2013},
      pages = {1--12},
      publisher = {ACM},
      isbn = {978-1-4503-2378-9},
      doi = {10.1145/2503210.2503277}
    }

### Publications

1. Alexandru Calotoiu, David Beckingsale, Christopher W. Earl, Torsten Hoefler, Ian Karlin, Martin Schulz, Felix Wolf: Fast Multi-Parameter Performance Modeling. In Proc. of the 2016 IEEE International Conference on Cluster Computing (CLUSTER), Taipei, Taiwan, pages 172–181, IEEE, September 2016. [PDF](https://apps.fz-juelich.de/jsc-pubsystem/aigaion/attachments/fastmultiparam.pdf-f839eba376c6d61a8c4cab9860b6b3bf.pdf)

2. Marcus Ritter, Alexandru Calotoiu, Sebastian Rinke, Thorsten Reimann, Torsten Hoefler, Felix Wolf: *Learning Cost-Effective Sampling Strategies for Empirical Performance Modeling.* In Proc. of the 34th IEEE International Parallel and Distributed Processing Symposium (IPDPS), New Orleans, LA, USA, pages 884–895, IEEE, May 2020. [PDF](https://apps.fz-juelich.de/jsc-pubsystem/aigaion/attachments/ritter_ea_2020_ipdps.pdf-01cbe96f7a170aba7c7ef941f966d292.pdf)

3. Marcus Ritter, Alexander Geiß, Johannes Wehrstein, Alexandru Calotoiu, Thorsten Reimann, Torsten Hoefler, Felix Wolf: *Noise-Resilient Empirical Performance Modeling with Deep Neural Networks.* In Proc. of the 35th IEEE International Parallel and Distributed Processing Symposium (IPDPS), Portland, Oregon, USA, pages 23–34, IEEE, May 2021. [PDF](http://htor.inf.ethz.ch/publications/img/noiseresilientmodeling.pdf)
