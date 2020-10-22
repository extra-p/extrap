Extension Points
================

Modelers
--------

Modelers are automatically registered and available for modeling if they are added to the correct location.
To add a new modeler to Extra-P,  you must add a new python module to the corresponding subpackage of `extrap.modelers`. 

Your new module must contain a class with the implementation of the new modeler. 
The class must inherit from `extrap.modelers.abstract_modeler.AbstractModeler`. It must, furthermore, include a 
class-level field `NAME` that contains a unique display name for the modeler.
Users can select the modeler using that name. 
You should also provide a class-level field `DESCRIPTION` with a short description of the modeler.
The description will be shown to the user in both the GUI and the CLI.

The most important method of the modeler is the `model` method. 
It is called with the parameter `measurements`, a sequence of measurement point sequences. 
It must create a model for each sequence of measurement points. 
The optional second parameter `progress_bar` is intended for progress reporting using the `tqdm` package.
The return value must be a Sequence of models. The index of each model  must correspond to the index of the 
measurement point sequence in `measurements`, which was used to create the model.


### Single-parameter modeler specifics

If you add a modeler for single-parameter models, add your module to the package `extrap.modelers.single_parameter`.

If your modeler only supports creating a single model at a time, you should extend `extrap.modelers.abstract_modeler.SingularModeler`.
Then you should not implement the `model` method. Instead, you must implement the `create_model` method.
The method is called with the parameter `measurements`, a sequence of measurement points. 
It must return a Model that should be based on the measurement points.

### Multi-parameter modeler specifics

To add a multi-parameter modeler, you have to create a module in `extrap.modelers.multi_parameter`.
All multi-parameter modelers must inherit from `extrap.modelers.abstract_modeler.MultiParameterModeler`.

They have a field called `single_parameter_modeler` containing a single-parameter modeler. 
This modeler should be used to create single-parameter models if the multi-parameter modeler needs them. 

Similar to single-parameter modelers, if your modeler only supports creating a single model at a time, 
your modeler should inherit `extrap.modelers.abstract_modeler.SingularModeler`. 
Note that `SingularModeler` should be used as mixin: `class MyMultiModeler(MultiParameterModeler, SingularModeler)`.
Furthermore, you must implement the `create_model` method.

### Modeler options

Options for the different modelers are handled via the `extrap.modelers.modeler_options` module.
It takes care of serializing and deserializing options when saving an experiment.
Furthermore, it provides an automatic generation of matching command-line options and GUI controls.

To enable options for your modeler, you must annotate your modeler class with `@modeler_options`.
To declare an option, you must use a class-level field which is assigned the result of `modeler_options.add(...)`
Use the instance field normally to get or set a value for an option.
Options may also be grouped using `modeler_options.group()`.
