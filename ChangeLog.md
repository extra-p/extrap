# Extra-p ChangeLog

***

## Released version 4.2

### Measurement point suggestion via GPR
* The GUI features a view for measurment point suggestion
* Several suggestion approaches help the user to complete or extend the set of available measurement points
to further improve model accuracy, improve the noise resilience of the model, while optimizing modeling cost.

***

## Released version 4.1

### Improved GUI
* Different views for the call-tree
* Function parameters can be shortened
* Automatic update check

### New formatting options
* Format for text output can be specified by the user 
  (see [docs/output-formatting.md](docs/output-formatting.md))
* Plots have formatting options for font family and font size

### Extended features
* New extendable File Reader API
* Unified extension loader
* Unified entities with names
* Added support for tags on entities

### Improved OS support
* Added support for macOS Ventura

### Upgraded dependencies
* Upgrade from PySide2 to PySide6
* Upgrade from Matplotlib 3.2 to Matplotlib 3.5
* Upgrade from PyObjC 6.2 to PyObjC 9.0
* Added support for pyCubexR 2.0 for faster reading of CUBEX files

***

## Released version 4.0
### Sparse modeling support
* Added support for sparse modeling for multi-parameter models. This allows the 
model generator to be run to use fewer measurement points.
* Added several strategies for the sparse modeler that specify how the 
measurement points will be used for modeling. Advanced users can use this option
to exactly specify which, and how many points should be used for modeling.

### Support for more input files
* Added two new input formats. Besides a text and the cubex-based file 
format Extra-P now also supports a JSON lines style format and a JSON
format. With the JSON lines style format experiment results can be written
consecutively into one file, that can then be used for modeling, allowing 
for more flexibility in case it is not clear how many or with which configuration
the experiments will be done.

### A python-based version of Extra-P
* The new version of Extra-P is completely Python-based and can be installed
more easily using the pip package manager. Besides some standard python packages,
there are no further dependencies anymore.

### Improved Python-based command-line tool
* Added a new Python-based command-line tool that allows accessing all of the
the functionality of Extra-P (besides the graphical plots).

### Minor improvements and fixes
* Fixed some GUI bugs
* Improved the readability of the GUI, introducing loading indicators for time
intensive operations
* Added new save file format for Extra-P

***

## Released version 3.0

### Command line tools

* Added the command line tool extrap-modeler to create a Extra-P
  experiment file with input data and models on a shell.
* Refactored extrap-print to produce a console output of an Extra-P
  experiment file. Removed the 95% confidence interval from the
  extrap-print output. Added additional options to get a list of
  definitions.

### Multi-parameter support

* Added possibility to open set of CUBE files with multiple
  parameters. The number of parameters can be specified before loading
  the CUBE files. After a change to the number of parameters, it tries to
  guess the structure of the measurement directories.
* Added a model generator for multi-parameter models. Depending whether
  the experiment contains only one parameter or multiple parameters, the
  single-parameter model generators or the multi-parameter model generator
  can be selected for model generation.
* Several new displays for two-parameter models:
    * A display that plots selected models next to each other in distinct
      surface plots
    * A display that plots all selected models in the same surface plot
    * A scatter plot that shows for each parameter only the dominating
      function
    * The dominating model as the surface plot
    * The dominating model as a heat map
    * contour plots
    * interpolated contour plots
    * All selected models in the same surface plot with transparent
      surface and measurement points.
* Added projection possibilities. Thus, when more parameters are available
  than the display can handle, it shows a orthogonal projection.
  The user can choose which parameter is shown on which axis and the value
  for the remaining parameters.

### Minor improvements and fixes

* Fixed Opening of experiment files which contain a model generated
  with the refining model generator.
* Allow to pass an Extra-P experiment file to the GUI via command line
  argument. The file is then opened on startup of the GUI.

***

## Released version 2.0

* Created custom GUI in python which replaces the usage of the
  CUBE GUI with plug-ins
* Added command line tool to output the content of Extra-P files.
* A refactored version of the single parameter modeler from version 1.0.
  It requires a manual definition of the model search space
* Added a new modeler that iteratively refined its modeling space and,
  thus, do not need a manual configured modeling space beforehand.
* Added a custom format to load and store perfromance models.
* Support for data input via a set of CUBE files contained in a directory.
  The measurement directories containing the CUBE files need to have
  a uniform format. The GUI tries to automatically detect the prefix name,
  the name of the parameter, the parameter values and the number of
  repetitons.
* Improved robustness against calltree variations during CUBE data import.
* Support for data input in a human readable text format.
* An experiment can contain multiple models. The user can switch between
  the models, create additional models or delete models.

***

## Released version 1.0

* First official release of extra-p library and executable
* First official release of Extrapolation plugin for Cube 4.3+
* First official release of Modelling plugin for Cube 4.3+
