Developer Quick Start
=====================
This guide will show you the steps to set up your development environment for Extra-P.

> ##### IMPORTANT
> If you plan to extend Extra-P in any way, please, read and follow the documentation on [extension points](extension-points.md).

### Setup an virtual environment to test/develop the package

1. `python -m venv venv` Create a new virtual python environment to test/develop the code.
2. Activate the virtual environment to use it for testing/developing.
    * On Windows, use `venv\Scripts\activate.bat`
    * On Unix or macOS, use `source venv/bin/activate`
3. `deactivate` Deactivate the virtual environment.

### Install the Extra-P package

#### Installing from a local src tree

`pip install -e <path>` installs package in developer mode from a local src tree via sym links. If you are already in
the root folder, you can use `pip install -e .`

#### Installing from PyPI

`python -m pip install extrap --upgrade` installs the Extra-P package. The `--upgrade` forces the installation of a new
version if a previous version is already installed.

### Run Extra-P

Run `extrap` to start the command line version of Extra-P. You can find several example datasets in
the [tests/data](../tests/data) folder. More about the usage of Extra-P can be found in
the [Quick Start Guide](quick-start.md). If you want to modify or extend Extra-P, please have a look at
the [extension points documentation](extension-points.md).

### Run Extra-P tests

The tests in the [tests](../tests) folder can be run with Python's unittest module or with PyTest. We recommend using
PyTest to execute the tests in Extra-P.

#### PyTest

1. `cd tests` change your working directory to the tests folder
2. Run `pytest` to start the test execution
    * By default, the tests need an installed GUI environment. If you want to omit the GUI tests (e.g. for CI) you can
      use the following option: `--ignore-glob=test_gui*.py`

#### Python unittest module

The following steps are necessary to use the unittest module.

1. Add the root folder to the `PYTHONPATH`
2. Change your working directory to the `tests` folder
3. Run the unittest module to start the test execution `python -m unittest`

On Windows, you can use the following command `set PYTHONPATH=%CD% & cd tests & python -m unittest`

### Build and publish Extra-P package

1. Make sure that the `__version__` variable in [extrap/\_\_init\_\_.py](../extrap/__init__.py) is updated.
2. `python setup.py sdist bdist_wheel` Create package from code.
3. `python -m twine upload dist/*` Upload package to python index. Need to specify username, password and do not forget
   to update the version of the package.

#### Upgrade commands for publishing tools:

* `python -m pip install --user --upgrade setuptools wheel`
* `python -m pip install --user --upgrade twine`
