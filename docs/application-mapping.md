Application Mapping
===================

For now the development of the application mapping is not finished. At the moment, only the basic functions for
application mapping are integrated into Extra-P.
This mainly refers to the model comparison functionality.

Model Comparison
----------------

You can use the GUI in Extra-P to compare two experiments
(an experiment means all models for each callpath and metric of a particular application).
For the comparison, Extra-P requires that these models share the same parameters
and that these parameters have the same unit of measurement.
The parameter values are allowed to differ.

The comparison only works in the GUI, therefore please start by opening the first experiment in the GUI.
Then you should select the *Compare with experiment* entry in the *File* menu.
An assistant is opened that guides you through the comparison process.
After the comparison process finishes, the GUI displays the common call-tree of your two experiments.
By default, each node in the call-tree shows both models.
These models are aggregated to include child-nodes that are not present in both sets of models.
The *[Comparison]* nodes in the calltree, contain the original models and call-tree parts for the corresponding
call-path.

Interactive Analysis
--------------------

Each entry in the call-tree has an annotation that shows which of the two models is the larger one at a given point.
You can choose this point via the sliders below the call-tree view.

In addition to the annotation, you can also select the *Comparison plot* in the *Plots* menu.
The comparison plot shows you the distance between the selected two models depending on the parameters.