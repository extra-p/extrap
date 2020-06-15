# TODOs for Extra-P 4.0 Release:

* Cubelib in python implementieren
* Delete metrics like min_, max_time from the beginning (Not relevant for our modeling)
* Don't model data where a metric is 0
* upgrade extrap project organization so we can use private repositories (Felix)
* Cubelib integration into extrap, implementation of the necessary functionality into python code (could be Student Group, PPT Lab Project...)
* CI/CD including unit tests, auto package build, when passing auto upload to pypi index with version id upgrade and change transcript. Could be done using a V-server from the university (could be Student Group, PPT Lab Project...)
* lock main branch, no commits to this branch, only over merge with approval from other branch (Marcus)
* gui version of extrap installable as additional feature only with setuptools (Marcus)
* New Output Format for Extra-P (Marcus)
* Readme File
* command line tool explanation with options
* Some kind of documentation slides with a complete work though the functionality of extrap
* Design Dokument
  * Requirements
  * Designs
  * Input/Output format
  * Modeler specifications
  * GUI functionality
  * Weak/Strong scaling support
  * Classes and their functions
  
  # TODO List:

* fix error in functions with too much coefficients in multi parameter case...
* gui version of extrap installable as additional feature only with setuptools
* make the GUI mask for cube files auto detect values smarter 
* New Extra-P Data format to save progress
* works with exponent refinement, Refinement modeler
* Test the tool against old results and fine tune the hypothesis computations
* the gui and its features work completely
* logging for gui and command line tool
* improve output from the command line tool so it is readable
* cube file reader in python code

# Ideas:

* no more PMNF, solve for 4PL logistic regression instead
* coefficients and exponents should be rounded for the representation in the gui and command line for the final result
* cube file reader:
  * make progress bar for command line tool
  * flag for the number of reps to be used
  * flag for the coords to be used
* text file reader:
  * weak, strong scaling flag also works (not sure if that is necessary at all?)
* a progress bar could be nice for all readers in general
