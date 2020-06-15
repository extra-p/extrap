"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.multi_parameter_modeler import MultiParameterModeler
from entities.single_parameter_modeler import SingleParameterModeler


class ModelGenerator:


    def __init__(self, experiment):
        self.experiment = experiment
        self.name = "New Modeler"
        self.modeler_id = self.experiment.get_new_modeler_id()
        # choose the modeler based on the input data
        self.modeler = self.choose_modeler()
        
        
    def choose_modeler(self):
        if len(self.experiment.parameters) == 1:
            # choose single parameter model generator when only one parameter
            return SingleParameterModeler(self.experiment, self.modeler_id, self.name)
        else:
            # choose multi parameter model generator when more than one parameter
            return MultiParameterModeler(self.experiment, self.modeler_id, self.name)
    
    
    def model_all(self, median):
        for callpath_id in range(len(self.experiment.callpaths)):
            for metric_id in range(len(self.experiment.metrics)):
                self.modeler.create_model(callpath_id, metric_id, median)
        # add the modeler with the results to the experiment
        self.experiment.add_modeler(self.modeler)
        return self.experiment
    
    
    def set_name(self, name):
        self.name = name
        
        
    def get_name(self):
        return self.name
        
        
    def get_id(self):
        return self.id
    
    