"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


from entities.compound_term import CompoundTerm
from entities.single_parameter_hypothesis import SingleParameterHypothesis
from entities.single_parameter_function import SingleParameterFunction
from entities.constant_function import ConstantFunction
from entities.constant_hypothesis import ConstantHypothesis
from entities.model import Model
from entities.single_parameter_modeler import SingleParameterModeler
from entities.multi_parameter_function import MultiParameterFunction
from entities.multi_paramter_term import MultiParameterTerm
from entities.compound_term import CompoundTermParameterPair
from entities.multi_parameter_hypothesis import MultiParameterHypothesis
import logging
import copy
from pip._internal.cli.cmdoptions import retries


class MultiParameterModeler:
    """
    This class represents the modeler for single parameter functions.
    In order to create a model measurements at least 5 points are needed.
    The result is either a constant function or one based on the PMNF.
    """


    def __init__(self, experiment, modeler_id, name):
        """
        Initialize SingleParameterModeler object.
        """
        self.experiment = experiment
        
        self.modeler_id = modeler_id
        
        self.name = name
        
        self.models = []
        
        # value for the minimum number of measurement points required for modeling
        self.min_measurement_points = 5
        
        # use mean or median measuremnt values to calculate models
        self.median = None


    def compare_parameter_values(self, parameter_value_list1, parameter_value_list2):
        """
        This method compares the parameter values of two coordinates with each other
        to see if they are equal and returns a True or False.
        """
        if len(parameter_value_list1) != len(parameter_value_list2):
            return False
        for i in range(len(parameter_value_list1)):
            if parameter_value_list1[i] != parameter_value_list2[i]:
                return False
        return True
    
    
    def get_parameter_values(self, coordinate, parameter_id, parameters):
        """
        This method returns the parameter values from the coordinate.
        But only the ones necessary for the compare_parameter_values() method.
        """
        parameter_value_list = []
        for i in range(len(parameters)):
            if i != parameter_id:
                _, value = coordinate.get_parameter_value(i)
                parameter_value_list.append(float(value))
        return parameter_value_list

    
    def find_first_measurement_points(self):
        """
        This method returns the ids of the coordinates that should be used for creating
        the single parameter models.
        """
        coordinate_lists = []
        coordinates = self.experiment.get_coordinates()
        parameters = self.experiment.get_parameters()
    
        for parameter_id in range(len(parameters)):
            parameter = parameters[parameter_id]
            coordinate_ids = []
            done = False
            
            for coordinate_id in range(len(coordinates)):
                reference_coordinate = coordinates[coordinate_id]
                parameter_value_list = self.get_parameter_values(reference_coordinate, parameter_id, parameters)
                    
                for coordinate_id2 in range(len(coordinates)):
                    test_coordinate = coordinates[coordinate_id2]
                    parameter_value_list2 = self.get_parameter_values(test_coordinate, parameter_id, parameters)
                    
                    #print(parameter_value_list)
                    #print(parameter_value_list2)
                    equal = self.compare_parameter_values(parameter_value_list, parameter_value_list2)
                    #print(equal)
                    
                    if equal == True:
                        coordinate_ids.append(coordinate_id2)
                #print(coordinate_ids)
                        
                if len(coordinate_ids) < self.min_measurement_points:
                    coordinate_ids.clear()
                    
                else:
                    done = True
                    break
                
            if done == True:
                coordinate_lists.append(coordinate_ids)
                
            else:
                logging.error("Not enough measurement points for parameter "+str(parameter.get_name())+".")
                
        return coordinate_lists

  
    def create_model(self, callpath_id, metric_id, median):
        """
        Create a model for the given callpath and metric using the given data.
        """
        # set to use mean or median measurement values
        self.median = median
        
        # use the first base points found for each parameter for modeling for the single parameter functions
        coordinates_list = self.find_first_measurement_points()
        #print(coordinates_list)
        
        functions = []
        
        # model all single parmaeter experiments using only the selected points from the step before
        parameters = self.experiment.get_parameters()
        for parameter_id in range(len(parameters)):
            coordinates = coordinates_list[parameter_id]
            #print(coordinates)
            single_parameter_modeler_id = self.experiment.get_new_modeler_id()
            name = "Single Paramater Modeler for Parameter "+str(parameters[parameter_id].get_name())
            single_parameter_modeler = SingleParameterModeler(self.experiment, single_parameter_modeler_id, name, coordinates)
            single_parameter_modeler.create_model(callpath_id, metric_id, median)
            model = single_parameter_modeler.get_model(callpath_id, metric_id)
            hypothesis = model.get_hypothesis()
            function = hypothesis.get_function()
            #print(function.to_string(parameters[parameter_id]))
            functions.append(function)
            
        # select the measurements by callpath_id and metric_id
        all_measurements = self.experiment.get_measurements()
        measurements = []
        for measurement_id in range(len(all_measurements)):       
            if all_measurements[measurement_id].get_callpath_id() == callpath_id and all_measurements[measurement_id].get_metric_id() == metric_id:
                measurements.append(all_measurements[measurement_id])

        # check if the number of measurements satisfies the reuqirements of the modeler (>=5)
        if len(measurements) < self.min_measurement_points:
            logging.error("Number of measurements for each parameter needs to be at least 5 in order to create a performance model.")
            return None
        
        # get the coordinates for modeling
        coordinates = self.experiment.get_coordinates()
            
        # use all available additional points for modeling the multi parameter models
        constantCost = 0
        meanModel = 0
        
        for i in range(len(measurements)):
            if self.median == True:
                meanModel += measurements[i].get_value_median() / float(len(measurements))
            else:
                meanModel += measurements[i].get_value_mean() / float(len(measurements))
        for i in range(len(measurements)):
            if self.median == True:
                constantCost += (measurements[i].get_value_median() - meanModel) * (measurements[i].get_value_median() - meanModel)
            else:
                constantCost += (measurements[i].get_value_mean() - meanModel) * (measurements[i].get_value_mean() - meanModel)
           
        # find out which parameters should be deleted 
        compound_terms = []
        delete_params = []
        keep_params = []
        
        for i in range(len(functions)):
            function = functions[i]
            compound_terms = function.get_compound_terms()
            if len(compound_terms) > 0:
                compound_term = compound_terms[0]
                compound_terms.append(compound_term)
                keep_params.append(i)
            else:
                delete_params.append(i)
        
        # see if the function is constant
        if len(delete_params) == len(parameters):
            constant_function = ConstantFunction()
            constant_function.set_constant_coefficient(meanModel)
            constant_hypothesis = ConstantHypothesis(constant_function, self.median)
            constant_hypothesis.set_RSS(constantCost)
            constant_hypothesis.set_SMAPE(0.0)
            constant_hypothesis.set_AR2(0.0)
            constant_hypothesis.set_rRSS(0.0)
            return constant_hypothesis
        
        # in case is only one parameter, make a single parameter function
        elif (len(parameters) - len(delete_params)) == 1:
            multi_parameter_function = MultiParameterFunction()
            multi_parameter_term = MultiParameterTerm()
            parameter_id = keep_params[0]
            parameter = self.experiment.get_parameter(parameter_id)
            compound_term = compound_terms[0]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, compound_term)
            multi_parameter_term.add_compound_term_parameter_pair(compound_term_parameter_pair)
            multi_parameter_term.set_coefficient(compound_term.get_coefficient())
            multi_parameter_function.add_multi_parameter_term(multi_parameter_term)
            constant_coefficient = functions[keep_params[0]].get_constant_coefficient()
            multi_parameter_function.set_constant_coefficient(constant_coefficient)
            multi_parameter_hypothesis = MultiParameterHypothesis(multi_parameter_function, self.median)
            multi_parameter_hypothesis.compute_cost(measurements, coordinates)
            return multi_parameter_hypothesis
        
        # Remove unneccessary parameters
        for i in range(len(delete_params)):
            parameters.pop(delete_params[i])
      
        hypotheses = []
        
        # add Hypotheses for 2 parameter models
        if len(parameters) == 2:
            # create multiplicative multi parameter terms
            mult = MultiParameterTerm()
            for i in range(len(compound_terms)):
                compound_term = compound_terms[i]
                compound_term.set_coefficient(1)
                parameter = parameters[i]
                compound_term_parameter_pair = CompoundTermParameterPair(parameter, compound_term)
                mult.add_compound_term_parameter_pair(compound_term_parameter_pair)
                
            # create additive multi parameter terms
            add = []
            for i in range(len(compound_terms)):
                mpt = MultiParameterTerm()
                compound_term = compound_terms[i]
                compound_term.set_coefficient(1)
                parameter = parameters[i]
                compound_term_parameter_pair = CompoundTermParameterPair(parameter, compound_term)
                mpt.add_compound_term_parameter_pair(compound_term_parameter_pair)
                mpt.set_coefficient(1)
                add.append(mpt)
                
            # create multi parameter functions
            f1 = MultiParameterFunction()
            f2 = MultiParameterFunction()
            f3 = MultiParameterFunction()
            f4 = MultiParameterFunction()
            
            # create f1 function a*b
            f1.add_multi_parameter_term(mult)
            
            # create f2 function a*b+a
            f2.add_multi_parameter_term(add[0])
            f2.add_multi_parameter_term(mult)
            
            # create f3 function a*b+b
            f3.add_multi_parameter_term(add[1])
            f3.add_multi_parameter_term(mult)
            
            # create f4 function a+b
            f4.add_multi_parameter_term(add[0])
            f4.add_multi_parameter_term(add[1])
            
            # create the hypotheses from the functions
            mph1 = MultiParameterHypothesis(f1, self.median)
            mph2 = MultiParameterHypothesis(f2, self.median)
            mph3 = MultiParameterHypothesis(f3, self.median)
            mph4 = MultiParameterHypothesis(f4, self.median)
            
            # add the hypothesis to the list
            hypotheses.append(mph1)
            hypotheses.append(mph2)
            hypotheses.append(mph3)
            hypotheses.append(mph4)
        
        
        # add Hypotheses for 3 parameter models
        if len(parameters) == 3:
        
            # create multiplicative multi parameter terms
            
            # x*y*z
            mult = MultiParameterTerm()
            for i in range(len(compound_terms)):
                compound_term = compound_terms[i]
                compound_term.set_coefficient(1)
                parameter = parameters[i]
                compound_term_parameter_pair = CompoundTermParameterPair(parameter, compound_term)
                mult.add_compound_term_parameter_pair(compound_term_parameter_pair)
            
            # x*y
            mult_x_y = MultiParameterTerm()
            ct0 = compound_terms[0]
            ct0.set_coefficient(1)
            parameter = parameters[0]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct0)
            mult_x_y.add_compound_term_parameter_pair(compound_term_parameter_pair)
            ct1 = compound_terms[1]
            ct1.set_coefficient(1)
            parameter = parameters[1]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct1)
            mult_x_y.add_compound_term_parameter_pair(compound_term_parameter_pair)
            
            # y*z
            mult_y_z = MultiParameterTerm()
            ct2 = compound_terms[1]
            ct2.set_coefficient(1)
            parameter = parameters[1]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct2)
            mult_y_z.add_compound_term_parameter_pair(compound_term_parameter_pair)
            ct3 = compound_terms[2]
            ct3.set_coefficient(1)
            parameter = parameters[2]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct3)
            mult_y_z.add_compound_term_parameter_pair(compound_term_parameter_pair)
            
            # x*z
            mult_x_z = MultiParameterTerm() 
            ct4 = compound_terms[0]
            ct4.set_coefficient(1)
            parameter = parameters[0]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct4)
            mult_x_z.add_compound_term_parameter_pair(compound_term_parameter_pair)
            ct5 = compound_terms[2]
            ct5.set_coefficient(1)
            parameter = parameters[2]
            compound_term_parameter_pair = CompoundTermParameterPair(parameter, ct5)
            mult_x_z.add_compound_term_parameter_pair(compound_term_parameter_pair)
            
            # create additive multi parameter terms
        
            # x+y+z
            add = []
            
            for i in range(len(compound_terms)):
                mpt = MultiParameterTerm()
                compound_term = compound_terms[i]
                compound_term.set_coefficient(1)
                parameter = parameters[i]
                compound_term_parameter_pair = CompoundTermParameterPair(parameter, compound_term)
                mpt.add_compound_term_parameter_pair(compound_term_parameter_pair)
                mpt.set_coefficient(1)
                add.append(mpt)
                
            # create multi parameter functions
            f0 = MultiParameterFunction()
            f1 = MultiParameterFunction()
            f2 = MultiParameterFunction()
            f3 = MultiParameterFunction()
            f4 = MultiParameterFunction()
            f5 = MultiParameterFunction()
            f6 = MultiParameterFunction()
            f7 = MultiParameterFunction()
            f8 = MultiParameterFunction()
            f9 = MultiParameterFunction()
            f10 = MultiParameterFunction()
            f11 = MultiParameterFunction()
            f12 = MultiParameterFunction()
            f13 = MultiParameterFunction()
            f14 = MultiParameterFunction()
            f15 = MultiParameterFunction()
            f16 = MultiParameterFunction()
            f17 = MultiParameterFunction()
            f18 = MultiParameterFunction()
            f19 = MultiParameterFunction()
            f20 = MultiParameterFunction()
            f21 = MultiParameterFunction()
            f22 = MultiParameterFunction()
        
            # x*y*z
            f1.add_multi_parameter_term(mult)
            
            # x+y+z
            f2.add_multi_parameter_term(add[0])
            f2.add_multi_parameter_term(add[1])
            f2.add_multi_parameter_term(add[2])
            
            # x*y*z+x
            f3.add_multi_parameter_term(mult)
            f3.add_multi_parameter_term(add[0])
            
            # x*y*z+y
            f4.add_multi_parameter_term(mult)
            f4.add_multi_parameter_term(add[1])
            
            # x*y*z+z
            f5.add_multi_parameter_term(mult)
            f5.add_multi_parameter_term(add[2])
            
            # x*y*z+x*y
            f6.add_multi_parameter_term(mult)
            f6.add_multi_parameter_term(mult_x_y)
            
            # x*y*z+y*z
            f7.add_multi_parameter_term(mult)
            f7.add_multi_parameter_term(mult_y_z)
            
            # x*y*z+x*z
            f8.add_multi_parameter_term(mult)
            f8.add_multi_parameter_term(mult_x_z)
            
            # x*y*z+x*y+z
            f9.add_multi_parameter_term(mult)
            f9.add_multi_parameter_term(mult_x_y)
            f9.add_multi_parameter_term(add[2])
            
            # x*y*z+y*z+x
            f10.add_multi_parameter_term(mult)
            f10.add_multi_parameter_term(mult_y_z)
            f10.add_multi_parameter_term(add[0])
            
            # x*y*z+x*z+y
            f0.add_multi_parameter_term(mult)
            f0.add_multi_parameter_term(mult_x_z)
            f0.add_multi_parameter_term(add[1])
            
            # x*y*z+x+y
            f11.add_multi_parameter_term(mult)
            f11.add_multi_parameter_term(add[0])
            f11.add_multi_parameter_term(add[1])
            
            # x*y*z+x+z
            f21.add_multi_parameter_term(mult)
            f21.add_multi_parameter_term(add[0])
            f21.add_multi_parameter_term(add[2])
            
            # x*y*z+y+z
            f22.add_multi_parameter_term(mult)
            f22.add_multi_parameter_term(add[1])
            f22.add_multi_parameter_term(add[2])
            
            # x*y+z
            f12.add_multi_parameter_term(mult_x_y)
            f12.add_multi_parameter_term(add[2])
            
            # x*y+z+y
            f13.add_multi_parameter_term(mult_x_y)
            f13.add_multi_parameter_term(add[2])
            f13.add_multi_parameter_term(add[1])
            
            # x*y+z+x
            f14.add_multi_parameter_term(mult_x_y)
            f14.add_multi_parameter_term(add[2])
            f14.add_multi_parameter_term(add[0])
            
            # x*z+y
            f15.add_multi_parameter_term(mult_x_z)
            f15.add_multi_parameter_term(add[1])
            
            # x*z+y+x
            f16.add_multi_parameter_term(mult_x_z)
            f16.add_multi_parameter_term(add[1])
            f16.add_multi_parameter_term(add[0])
            
            # x*z+y+z
            f17.add_multi_parameter_term(mult_x_z)
            f17.add_multi_parameter_term(add[1])
            f17.add_multi_parameter_term(add[2])
            
            # y*z+x
            f18.add_multi_parameter_term(mult_y_z)
            f18.add_multi_parameter_term(add[0])
            
            # y*z+x+y
            f19.add_multi_parameter_term(mult_y_z)
            f19.add_multi_parameter_term(add[0])
            f19.add_multi_parameter_term(add[1])
            
            # y*z+x+z
            f20.add_multi_parameter_term(mult_y_z)
            f20.add_multi_parameter_term(add[0])
            f20.add_multi_parameter_term(add[2])

            # create the hypotheses from the functions
            mph0 = MultiParameterHypothesis(f0, self.median)
            mph1 = MultiParameterHypothesis(f1, self.median)
            mph2 = MultiParameterHypothesis(f2, self.median)
            mph3 = MultiParameterHypothesis(f3, self.median)
            mph4 = MultiParameterHypothesis(f4, self.median)
            mph5 = MultiParameterHypothesis(f5, self.median)
            mph6 = MultiParameterHypothesis(f6, self.median)
            mph7 = MultiParameterHypothesis(f7, self.median)
            mph8 = MultiParameterHypothesis(f8, self.median)
            mph9 = MultiParameterHypothesis(f9, self.median)
            mph10 = MultiParameterHypothesis(f10, self.median)
            mph11 = MultiParameterHypothesis(f11, self.median)
            mph12 = MultiParameterHypothesis(f12, self.median)
            mph13 = MultiParameterHypothesis(f13, self.median)
            mph14 = MultiParameterHypothesis(f14, self.median)
            mph15 = MultiParameterHypothesis(f15, self.median)
            mph16 = MultiParameterHypothesis(f16, self.median)
            mph17 = MultiParameterHypothesis(f17, self.median)
            mph18 = MultiParameterHypothesis(f18, self.median)
            mph19 = MultiParameterHypothesis(f19, self.median)
            mph20 = MultiParameterHypothesis(f20, self.median)
            mph21 = MultiParameterHypothesis(f21, self.median)
            mph22 = MultiParameterHypothesis(f22, self.median)
            
            # add the hypothesis to the list
            hypotheses.append(mph0)
            hypotheses.append(mph1)
            hypotheses.append(mph2)
            hypotheses.append(mph3)
            hypotheses.append(mph4)
            hypotheses.append(mph5)
            hypotheses.append(mph6)
            hypotheses.append(mph7)
            hypotheses.append(mph8)
            hypotheses.append(mph9)
            hypotheses.append(mph10)
            hypotheses.append(mph11)
            hypotheses.append(mph12)
            hypotheses.append(mph13)
            hypotheses.append(mph14)
            hypotheses.append(mph15)
            hypotheses.append(mph16)
            hypotheses.append(mph17)
            hypotheses.append(mph18)
            hypotheses.append(mph19)
            hypotheses.append(mph20)
            hypotheses.append(mph21)
            hypotheses.append(mph22)
        
        # select one function as the bestHypothesis for the start
        best_hypothesis = copy.deepcopy(hypotheses[0])
        best_hypothesis.compute_coefficients(measurements, coordinates)
        best_hypothesis.compute_cost(measurements, coordinates)
        best_hypothesis.compute_adjusted_rsquared(constantCost, measurements)
        
        print("hypothesis 0 : "+str(best_hypothesis.get_function().to_string())+" --- smape: "+str(best_hypothesis.get_SMAPE())+" --- ar2: "+str(best_hypothesis.get_AR2())+" --- rss: "+str(best_hypothesis.get_RSS())+" --- rrss: "+str(best_hypothesis.get_rRSS())+" --- re: "+str(best_hypothesis.get_RE()))
 
        # find the best hypothesis
        for i in range(1, len(hypotheses)):
            hypotheses[i].compute_coefficients(measurements, coordinates)
            hypotheses[i].compute_cost(measurements, coordinates)
            hypotheses[i].compute_adjusted_rsquared(constantCost, measurements)
         
            print("hypothesis "+str(i)+" : "+str(hypotheses[i].get_function().to_string())+" --- smape: "+str(hypotheses[i].get_SMAPE())+" --- ar2: "+str(hypotheses[i].get_AR2())+" --- rss: "+str(hypotheses[i].get_RSS())+" --- rrss: "+str(hypotheses[i].get_rRSS())+" --- re: "+str(hypotheses[i].get_RE()))
 
            if hypotheses[i].get_SMAPE() < best_hypothesis.get_SMAPE():
                best_hypothesis = copy.deepcopy(hypotheses[i])
        
        # add the best found hypothesis to the model list
        model = Model(best_hypothesis, callpath_id, metric_id)
        self.models.append(model)
        
        print("best hypothesis: "+str(best_hypothesis.get_function().to_string())+" --- smape: "+str(best_hypothesis.get_SMAPE())+" --- ar2: "+str(best_hypothesis.get_AR2())+" --- rss: "+str(best_hypothesis.get_RSS())+" --- rrss: "+str(best_hypothesis.get_rRSS())+" --- re: "+str(best_hypothesis.get_RE()))
     
     
    def get_models(self):
        return self.models
    
    
    def get_model(self, callpath_id, metric_id):
        for model_id in range(len(self.models)):
            model = self.models[model_id]
            if model.get_callpath_id() == callpath_id and model.get_metric_id() == metric_id:
                return model
        return None
    
