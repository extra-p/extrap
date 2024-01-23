# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from extrap.entities.coordinate import Coordinate
import itertools
from itertools import product
import math
from math import log2
from extrap.gui.Utils import tryCastNumToInt, tryCastListToInt, tryCastTupelToInt

class Struct():
  def __init__(self, check: bool, val, msg: str, payload = None):
    self.check = check
    self.val = val
    self.msg = msg
    self.payload = payload
    

class GPR_Interface():
        
    def calculateSunkenCost(experiment, coreParameter_index, selectedCallpath = None) -> Struct:
        #Calculate the cost
        if experiment is None:
            return Struct(False, -1, "Experiment is None")
        if selectedCallpath != None:
            if len(selectedCallpath) > 1:
                return Struct(False, -1, "Select a single callpath.")
            if len(selectedCallpath) < 1:
                return Struct(False, -1, "Select one callpath.")
        
        corehourse_container = {}
        for (callpath, metric), measurements in experiment.measurements.items():
            if str(metric).lower() != "time" and str(metric).lower() != "runtime":
                continue
            corehourse = sum(measurement.mean * len(measurement.values) * measurement.coordinate[coreParameter_index] for measurement in measurements) 
            corehourse = round(corehourse)
            corehourse_perCoord = {measurement.coordinate: round(measurement.mean * len(measurement.values) * measurement.coordinate[coreParameter_index]) for measurement in measurements}
            corehourse_container[callpath] = corehourse_perCoord        
            
        if selectedCallpath != None:
            callpath = selectedCallpath[0].path
            corehourse = sum(list(corehourse_container[callpath].values())) 
            return Struct(True, corehourse, "CP: \t"+ str(callpath) +": "+ str(corehourse), payload = corehourse_container)

        corehourse = sum(sum(list(corehourse_perCoord.values())) for corehourse_perCoord in corehourse_container.values()) 
        return Struct(True, corehourse, "Total: \t"+ str(corehourse), payload = corehourse_container)
        
        
    def checkMinNumberOfPoints(experiment) -> Struct:        
        #Check for min number of measurments    
        min_points = 3 * len(experiment.parameters)
        if min_points > len(experiment.coordinates):
            return Struct(False, min_points, "Not enougth points measured, at least "+ str(min_points) +" needed.\nMeasure the cheapest line of 5 points per parameter.")
        return Struct(True, min_points, "")        
        
        
    def checkMaxNumberOfPoints(experiment) -> Struct:        
        #Check for max number of measurments    
        max_points = 5 ** len(experiment.parameters)
        if max_points <= len(experiment.coordinates):
            return Struct(False, max_points, "Expriment is fully measured.")
        return Struct(True, max_points, "")
        
        
    def checkFivePointLines(experiment) -> Struct:        
        #Check for row of five for every parameters   
        if len(experiment.parameters) < 2:
            if len(experiment.coordinates) > 4:
                return Struct(True, len(experiment.coordinates), "")
            else:
                return Struct(False, len(experiment.coordinates), "Five measurements in a line needed per parameter.\nParameter "+ str(experiment.parameters[0].name) +" is missing "+ str(5 - len(experiment.coordinates)) +" points.") 
        else:
            longestCoords = [(0, -1)] * len(experiment.parameters)
            for i_p, p in enumerate(experiment.parameters):
                paramValues_allOccurrences = list(map(lambda coord: coord[i_p], experiment.coordinates))
                paramValues_distinct = list(dict.fromkeys(paramValues_allOccurrences))
                paramValues_occurrencesCounts = list(map(lambda x: round(paramValues_allOccurrences.count(x)), paramValues_distinct))
                longestCoords[i_p] = (max(paramValues_occurrencesCounts), paramValues_distinct[paramValues_occurrencesCounts.index(max(paramValues_occurrencesCounts))]) 
            shortestLongestCoords = min(longestCoords, key=lambda item: item[0])
            if shortestLongestCoords[0] > 4:
                return Struct(True, shortestLongestCoords, "")
            else:
                parameterName = experiment.parameters[longestCoords.index(shortestLongestCoords)].name
                parameterValueSugestion = tryCastNumToInt(shortestLongestCoords[1])
                return Struct(False, shortestLongestCoords, "Five measurements in a line needed per parameter.\nParameter "+ parameterName +" is missing "+ str(5 - shortestLongestCoords[0]) +" points at "+ parameterName +" = "+ str(parameterValueSugestion) +".")


    def get_eval_string(function_string):
        function_string = function_string.replace(" ","")
        function_string = function_string.replace("^","**")
        function_string = function_string.replace("log2","math.log2")
        function_string = function_string.replace("+-","-")
        return function_string
        
    def calculatePotentialCost(experiment, function_string):
        paremeterValues = {}
        for i_p, p in enumerate(experiment.parameters):
            paramValues_allOccurrences = list(map(lambda coord: coord[i_p], experiment.coordinates))
            paramValues_distinct = list(dict.fromkeys(paramValues_allOccurrences))
            paremeterValues[p.name] = paramValues_distinct
            
        allPoints = list(paremeterValues.values())[0]
        costPredictions = []
        if len(paremeterValues.keys()) == 1: 
            raise NotImplementedError
        elif len(paremeterValues.keys()) == 2: 
            allPoints = list(itertools.product(list(paremeterValues.values())[0], list(paremeterValues.values())[1]))
            allNewPoints = [p for p in allPoints if Coordinate(p) not in experiment.coordinates]
            for p in allNewPoints:
                paramA = p[0]
                paramB = p[1]
                costPredictions.append((eval(function_string), p))
        elif len(paremeterValues.keys()) == 3: 
            allPoints = list(itertools.product(list(paremeterValues.values())[0], list(paremeterValues.values())[1], list(paremeterValues.values())[2]))
            allNewPoints = [p for p in allPoints if Coordinate(p) not in experiment.coordinates]
            for p in allNewPoints:
                paramA = p[0]
                paramB = p[1]
                paramC = p[2]
                costPredictions.append((eval(function_string), p))
        elif len(paremeterValues.keys()) == 4: 
            allPoints = list(itertools.product(list(paremeterValues.values())[0], list(paremeterValues.values())[1], list(paremeterValues.values())[2], list(paremeterValues.values())[3]))
            allNewPoints = [p for p in allPoints if Coordinate(p) not in experiment.coordinates]
            for p in allNewPoints:
                paramA = p[0]
                paramB = p[1]
                paramC = p[2]
                paramD = p[3]
                costPredictions.append((eval(function_string), p))
        else:
            raise NotImplementedError  

        costPredictions.sort(key=lambda p: p[0])
        return costPredictions
        
        
    def constructFunctionString(experiment, model):
        parameterPlaceholder = [""] * len(experiment.parameters)
        for i_p, p in enumerate(experiment.parameters):
            parameterPlaceholder[i_p] = "param"+ str(chr(65 + i_p))

        # get the extrap function as a string
        function_string = model.hypothesis.function.to_string(*parameterPlaceholder)
        function_string = GPR_Interface.get_eval_string(function_string)
        return function_string

        
    def adviceMeasurement(MeasurementWizardWidget) -> Struct: 
        experiment = MeasurementWizardWidget.main_widget.getExperiment()
        if experiment is None:
            return Struct(False, -1, "Experiment is no loaded.")
        
        if str(MeasurementWizardWidget.main_widget.getSelectedMetric()).lower() != "time" and str(MeasurementWizardWidget.main_widget.getSelectedMetric()).lower() != "runtime":
            return Struct(False, -1, "Metric is not time.")

        check_struct = GPR_Interface.checkMaxNumberOfPoints(experiment)
        if not check_struct.check:
            return check_struct 
            
        numProc_ParamIndex = MeasurementWizardWidget.processesParameter.currentIndex()
        if numProc_ParamIndex < 0 :
            return Struct(False, -1, "No processes parameter selected.")
        selected_callpath = MeasurementWizardWidget.main_widget.getSelectedCallpath()
        
        if len(selected_callpath) > 1:
            return Struct(False, -1, "Please select a single callpath.")
        if len(selected_callpath) < 1:
            return Struct(False, -1, "Please select a callpath.")
        
        check_struct = GPR_Interface.checkMinNumberOfPoints(experiment)
        if not check_struct.check:
            return check_struct 
            
        check_struct = GPR_Interface.checkFivePointLines(experiment)#check is missing, when crossing 9, if not 10
        if not check_struct.check:
            return check_struct

        sunkenCost_struct = GPR_Interface.calculateSunkenCost(experiment, numProc_ParamIndex, selectedCallpath=selected_callpath)

        modeler = MeasurementWizardWidget.main_widget.getCurrentModel()
        model = modeler.models[selected_callpath[0].path, MeasurementWizardWidget.main_widget.getSelectedMetric()]
        if model == None:
            return Struct(False, -1, "Model is none.")
        function_string = GPR_Interface.constructFunctionString(experiment, model)            
        
        costPredictions = GPR_Interface.calculatePotentialCost(experiment, function_string)
        bestPoints = '\n'.join([str(tryCastTupelToInt(p[1])) for p in costPredictions[:3]])
        return Struct(True, -1, bestPoints, payload = costPredictions[:3]) 



        #min zwei Linien mit 5 punkte, wenn die sich nicht schneiden dann 10, wenn doch dann nur 9
        #Wiederholungen 4 vorschlagen
        #Immer die kkosten des aktuellen CP verwenden, nie über kinder summieren
        #möglichen punkte ausdenken, alle kosten berechnen, den billigsten nehemn
        #Z. 536 
        #return Struct(True, -1, "Done"+ str(function_string))   
        