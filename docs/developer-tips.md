Developer Tips
===============

If you plan to extend Extra-P in any way, please, read and follow the documentation on [extension points](extension-points.md).

### Build Extra-P package

1. `python setup.py sdist bdist_wheel` Create package from code.
2. `python -m twine upload --repository testpypi dist/*` Upload package to python index. Need to specify username, password and do not forget to update the version of the package.

#### Build virtual env to test package

This command only works in windows shell...

1. `python -m venv /tmppython` Create a new virtual python environment to test the code.
2. `\tmppython\Scripts\activate` Activate the virtual environment to use it for testing.
3. `deactivate` Deactivate the virtual environment.

#### Install the Extra-P package

1. `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps extrap-meaparvitas --upgrade` Install the Extra-P package. The `--upgrade` forces the installation of a new version if a previous version is already installed.
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