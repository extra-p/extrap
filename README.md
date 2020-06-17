# Extra-P

![Python package](https://github.com/MeaParvitas/Extra-P/workflows/Python%20package/badge.svg?branch=master)

Example project to setup the basic python package and test integration.


## TOC:

1. [Requirements](#Requirements)
2. [How to use it](#Usage)
3. [Notes](#Notes)

## Requirements

* Python 3
* PyQt 5 (only if using GUI)
* maplotlib
* numpy

## Usage

* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt` Text files.
* `extrap --json --path C:\Users\Admin\git\Extra-P\data\json\input_1.JSON` JSON files.
* `extrap --talpas --path C:\Users\Admin\git\Extra-P\data\talpas\talpas_1.txt` Talpas files.
* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\one_parameter_1.txt --out C:\Users\Admin\Desktop\test.txt` Create model and save it to text file at the given path.
* ``
* ``

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

## Notes

One reason why the gui is not showing can be missing python packages!


# Build Extra-P package

1. `python setup.py sdist bdist_wheel` Create package from code.
2. `python -m twine upload --repository testpypi dist/*` Upload package to python index. Need to specify username, password and do not forget to update the version of the package.

# Build virtual env to test package

This command only works in windows shell...

1. `python -m venv /tmppython` Create a new virtual python environment to test the code.
2. `tmppython\Scripts\activate` Activate the virtual environment to use it for testing.
3. `deactivate` Deactivate the virtual environment.

# Install the Extra-P package

1. `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps extrap-meaparvitas --upgrade` Install the Extra-P package. The ``--upgrade` forces the installation of a new version if a previous version is already installed.
2. `extrap` To run the command line version of Extra-P.

# Installing from a local src tree

1. `pip install -e <path>` Install package from a local src tree via a sim link.

# Command Line stuff:

* PyQT needs to be install in order for the gui to display when running the extrapgui command in a terminal.
* python -m pip install --user --upgrade twine
* python -m pip install --user --upgrade setuptools wheel
* `pip install -e C:\Users\Admin\git\extrap\`
* `extrap --text --path C:\Users\Admin\git\Extra-P\data\text\two_parameter_1.txt`
* `extrap --cube --path C:\Users\Admin\git\Extra-P\data\cube\kripke\`
* `extrap --cube --path C:\Users\Admin\git\Extra-P\data\cube\blast\`

# TODO:

* ...

## C++ shared libraries

In order to create a C++ shared library and use it the relevant source code needs to be compiled with the following command `g++ -Wall -O3 -fPIC -I/home/username/Cube/Cubelib/include/cubelib -shared CubeInterface.cc -o CubeInterface.so`. It is important to specify the correct path to the Cube library that is installed on the system, as the Cube Interface requires this library. The path shown here `/home/user/Cube/Cubelib/include/cubelib` is just an example and can vary depending on where you installed Cube on your system. Furthermore, to find the `#includes` all C++ files need to be mentioned when compiling the shared library like so `g++ -Wall -O3 -fPIC -I/home/marcus/Cube/Cubelib/include/cubelib -shared HelperClass.cc MainClass.cc -o MainClass.so`.

## Commands

### Python commands for testing

`python3 extrap.py --cube --path /home/marcus/GitLab/extrap/testdata/cube/blast`

### Compile the test example for the shared extrap library

`g++ -Wall -O3 -fPIC -shared -I/home/marcus/Cube/Cubelib/include/cubelib MessageStream.cc IoHelper.cc Utilities.cc Parameter.cc Printer.cc Test.cc -o Test.so -L/home/marcus/Cube/Cubelib/lib -lcube4`

### Compile the `CubeInterface.dll` C++ shared library

`g++ -Wall -O3 -fPIC -I/home/marcus/Cube/Cubelib/include/cubelib -shared Types.h Callpath.cc CompoundTerm.cc Coordinate.cc CubeMapping.cc DataPoint.cc Experiment.cc ExperimentPoint.cc Fraction.cc Function.cc IncrementalPoint.cc IoHelper.cc MessageStream.cc Metric.cc Model.cc ModelComment.cc ModelGenerator.cc ModelGeneratorOptions.cc MultiParameterFunction.cc MultiParameterFunctionModeler.cc MultiParameterHypothesis.cc MultiParameterModelGenerator.cc MultiParameterSimpleFunctionModeler.cc MultiParameterSimpleModelGenerator.cc MultiParameterSparseFunctionModeler.cc MultiParameterSparseModelGenerator.cc MultiParameterTerm.cc Parameter.cc Region.cc SimpleTerm.cc SingleParameterExhaustiveFunctionModeler.cc SingleParameterExhaustiveModelGenerator.cc SingleParameterFunction.cc SingleParameterFunctionModeler.cc SingleParameterHypothesis.cc SingleParameterModelGenerator.cc SingleParameterRefiningFunctionModeler.cc SingleParameterRefiningModelGenerator.cc SingleParameterSimpleFunctionModeler.cc SingleParameterSimpleModelGenerator.cc Utilities.cc CubeInterface.cc -o CubeInterface.dll -L/home/marcus/Cube/Cubelib/lib -lcube4 -w`

### Compile the `CubeInterface.so` C++ shared library

`g++ -Wall -O3 -fPIC -I/home/marcus/Cube/Cubelib/include/cubelib -shared Types.h Callpath.cc CompoundTerm.cc Coordinate.cc CubeMapping.cc DataPoint.cc Experiment.cc ExperimentPoint.cc Fraction.cc Function.cc IncrementalPoint.cc IoHelper.cc MessageStream.cc Metric.cc Model.cc ModelComment.cc ModelGenerator.cc ModelGeneratorOptions.cc MultiParameterFunction.cc MultiParameterFunctionModeler.cc MultiParameterHypothesis.cc MultiParameterModelGenerator.cc MultiParameterSimpleFunctionModeler.cc MultiParameterSimpleModelGenerator.cc MultiParameterSparseFunctionModeler.cc MultiParameterSparseModelGenerator.cc MultiParameterTerm.cc Parameter.cc Region.cc SimpleTerm.cc SingleParameterExhaustiveFunctionModeler.cc SingleParameterExhaustiveModelGenerator.cc SingleParameterFunction.cc SingleParameterFunctionModeler.cc SingleParameterHypothesis.cc SingleParameterModelGenerator.cc SingleParameterRefiningFunctionModeler.cc SingleParameterRefiningModelGenerator.cc SingleParameterSimpleFunctionModeler.cc SingleParameterSimpleModelGenerator.cc Utilities.cc CubeInterface.cc -o CubeInterface.so -L/home/marcus/Cube/Cubelib/lib -lcube4 -w`

The `-w` flag an be used to disable the warnings of the compiler.

### Notes

* The current cube interface implementation only supports 3 parameters
* Float values in the cube files are automatically detected but they need to be formatted like this "p4.s1000.t0,1.r1"
