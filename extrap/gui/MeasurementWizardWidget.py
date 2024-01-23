# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import numpy as np
from extrap.entities.callpath import Callpath
from extrap.entities.metric import Metric
import sys

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import *  # @UnusedWildImport
from PySide6.QtGui import QDoubleValidator

import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from extrap.gui.components.ExpanderWidget import ExpanderWidget
from extrap.gui.GPR_Interface import GPR_Interface, Struct
from extrap.gui.Utils import tryCastListToInt

class HistogrammWidget(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(dpi=85, figsize=(5, 2), facecolor="#bebebe")
        super(HistogrammWidget, self).__init__(fig) 
        self._axes = fig.add_subplot(111)
        self._axes.margins(tight=True)
        
    def setData(self, x, y):
        self.reset()
        self._axes.bar(x, y, width=0.9, color="#2277b0", edgecolor="black")
        self.y = y
        self.x = x
        self.draw() #trigger to rerender plot
        
    def addData(self, stackedY):
        if len(self.y) == len(stackedY):
            self._axes.bar(self.x, stackedY, width=0.9, color="green", edgecolor="black", bottom=self.y)
            self.y = [sum(i) for i in zip(self.y, stackedY)]
            self.draw() #trigger to rerender plot
      
    def reset(self):
        self._axes.clear()
        
        
class ParameterCourse_Manual(QWidget):
    def __init__(self):
        super(ParameterCourse_Manual, self).__init__()
        grid = QGridLayout(self)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setContentsMargins(11, 3, 11, 7)
        grid.setSpacing(7)
        self.setLayout(grid)

        
class ParameterWidget(QWidget):
    def __init__(self):
        super(ParameterWidget, self).__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(11, 3, 11, 7)
        grid.setSpacing(0)
        self.setLayout(grid)
        
        #Charactersitics of the parameter 
        self.parameterShape = ParameterCourse_Manual()
        grid.addWidget(self.parameterShape, 0, 0, 1, 2)        
           
        #Histogramm of parameter values
        self.histogram = HistogrammWidget()
        self.histogram.setToolTip("Historgamm over the measured values of this parameter.")
        grid.addWidget(self.histogram, 1, 0, 1, 2)
        grid.setRowMinimumHeight(1, 25)
        grid.setColumnMinimumWidth(0, 130)
        grid.setColumnMinimumWidth(1, 130)
    
    def reset(self):
        self.histogram.reset()
        

class MeasurementWizardWidget(QWidget):
    
    def __init__(self, mainWidget, parent):
        super(MeasurementWizardWidget, self).__init__(parent)
        self.main_widget = mainWidget
        self.main_layout = QHBoxLayout(self)
        self.init_empty()

    def init_empty(self):
        self.setLayout(self.main_layout)
        self.empty_label = QLabel(self)
        self.empty_label.setText("Please load a performance experiment.")
        self.main_layout.addWidget(self.empty_label)

    def init_single_parameter(self):
        self.main_layout.removeWidget(self.empty_label)
        self.empty_label.deleteLater()
        self.empty_label = None

        layout = QGridLayout(self)
        layout.setRowStretch(99, 1)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)

        self.main_layout.addLayout(layout)

        label = QLabel(self)
        label.setText("Dummy Text")
        layout.addWidget(label, 0, 0)

        modeling_budget = QLineEdit(self)
        modeling_budget.setValidator(QDoubleValidator())
        layout.addWidget(modeling_budget, 0, 1)

        #self.setLayout(grid)
        """grid.addWidget(QWidget(), 99, 0)
        grid.setRowStretch(99, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        label = QLabel(self)
        label.setText("Dummy Text")
        grid.addWidget(label, 0, 0)"""

    def init(self):    
        maxNumberOfParameters = 4
        grid = QGridLayout(self)
        self.setLayout(grid)
        grid.addWidget(QWidget(), 99, 0)
        grid.setRowStretch(99, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
            
        #Row Counter
        row = 0
                
        #Budget for improvment
        label = QLabel(self)
        label.setText("Modeling budget (in Core hours)")
        label.setToolTip("The modeling budget amounts to the total cost you are willing to spend for conducting performance measurements of your application calculated in core hours.")
        self._experimentBudget = QSpinBox(self)
        self._experimentBudget.setMinimum(0)
        self._experimentBudget.setMaximum(2147483647)
        self._experimentBudget.setValue(0)
        self._experimentBudget.valueChanged.connect(self.calculateSunkenCost)
        grid.addWidget(label, row, 0)
        grid.addWidget(self._experimentBudget, row, 1)
        row +=1
                
        #Processes parameter
        label = QLabel(self)
        label.setText("# Processes Param.")
        label.setToolTip("Please select which parameter represents the number of processes.")
        self.processesParameter = QComboBox(self)
        self.processesParameter.setToolTip("Please select which parameter represents the number of processes.")
        self.processesParameter.currentIndexChanged.connect(self.calculateSunkenCost)
        grid.addWidget(label, row, 0)
        grid.addWidget(self.processesParameter, row, 1)
        row +=1
                
        #Sunken Cost
        label = QLabel(self)
        label.setText("Sunken Cost [ch]")
        label.setToolTip("Number of core hourse used so far in this experiment.")
        self._sunkenCost = QLabel(self)
        self._sunkenCost.setToolTip("Number of core hourse used so far in this experiment. This can only be calculated when the time is measured.")
        grid.addWidget(label, row, 0, Qt.AlignTop)
        grid.addWidget(self._sunkenCost, row, 1)
        row +=1
		
        #Parameter Limitations        
        self._parameters = []
        for i in range(maxNumberOfParameters):
            rowCorrection = i*0
            
            parameterInfo = ParameterWidget()
            parameterOptCon = ExpanderWidget(self, "")
            parameterOptCon.setContent(parameterInfo)
            parameterOptCon.setToolTip("Please select for every Parameter a min and a max value and also a course. <br>These values are used for the generation of the new measurement points.")
            grid.addWidget(parameterOptCon, row +rowCorrection, 0, 1, 2)
            row +=1
               
            self._parameters.append((parameterInfo, parameterOptCon))
        rowCorrection = maxNumberOfParameters*2
        
        #Advise future measurement
        self._adviseMeasurement= QPushButton(self)
        self._adviseMeasurement.setText("Advise measurements")
        self._adviseMeasurement.setToolTip("Advise me future measurement points.")
        self._adviseMeasurement.clicked.connect(self.adviceMeasurment)
        grid.addWidget(self._adviseMeasurement, row +rowCorrection, 0)
        self._adviseStrategy = QComboBox(self)
        self._adviseStrategy.addItem("Generic")#sparse
        self._adviseStrategy.addItem("Hybrid")
        grid.addWidget(self._adviseStrategy, row +rowCorrection, 1)
        row +=1
        
        #Advice Labels
        self._adviseMeasurementLabel = QLabel(self)
        grid.addWidget(self._adviseMeasurementLabel, row +rowCorrection, 0, 1, 2, alignment=Qt.AlignCenter)
        row +=1
        self._adviseMeasurementPoints = QLabel(self)
        self._adviseMeasurementPoints.setWordWrap(True)
        self._adviseMeasurementPoints.setAlignment(Qt.AlignRight)
        grid.addWidget(self._adviseMeasurementPoints, row +rowCorrection, 0)
        self._adviseMeasurementStats = QLabel(self)
        grid.addWidget(self._adviseMeasurementStats, row +rowCorrection, 1, 2, 1)  
        row +=1  
   

    def reset(self, model_parameters=1):
    
        # if there is only one model parameter
        if model_parameters == 1:
            self.init_single_parameter()
            self.repaint()
            self.update()
            #TODO:
        
        # if there are several model parameters
        else:
            pass
            #TODO: fix this stuff
            """#Budget for improvment
            self._experimentBudget.setEnabled(False)
                    
            #Processes parameter
            self.processesParameter.setEnabled(False)
            self.processesParameter.setCurrentIndex(-1)
            
            #Sunken Cost
            self._sunkenCost.setText("")
            
            #Limitations for parameters        
            for i, parameter in enumerate(self._parameters):
                parameter[0].reset()
                parameter[1].setTitle("Parameter")
                parameter[1].setEnabled(False)
                parameter[1].toggle(False)
                if i >= model_parameters:
                    parameter[1].setVisible(False)
                else:
                    parameter[1].setVisible(True)
            
            #Advise future measurement
            self._adviseMeasurement.setEnabled(False)  
            self._adviseStrategy.setEnabled(False)
            
            #Advice Labels
            self._adviseMeasurementLabel.setText("")
            self._adviseMeasurementPoints.setText("")
            self._adviseMeasurementStats.setText("")"""
        
    
    def experimentChanged(self):
        
        experiment = self.main_widget.getExperiment()
        
        if experiment is None:
            self.reset()
            return
        
        self.reset(len(experiment.parameters))
        
        #Budget for improvment
        #self._experimentBudget.setEnabled(True)
                
        """#Processes parameter
        self.processesParameter.clear()
        for i, p in enumerate(experiment.parameters):
            self.processesParameter.addItem(str(p))
            self._parameters[i][1].setTitle("Parameter "+ str(p))
        self.processesParameter.setEnabled(True)
        if len(experiment.parameters) == 1:
            self.processesParameter.setCurrentIndex(0)
        else:
            self.processesParameter.setCurrentIndex(-1)
        
        #Sunken Cost
        self.calculateSunkenCost()
            
        #Limitations for parameters                   
        for parameterIndex, parameter in enumerate(self._parameters): 
            if parameterIndex >= numberOfParameters:
                break
            paramValues_allOccurrences = tryCastListToInt(list(map(lambda coord: coord[parameterIndex], experiment.coordinates)))
            paramValues_distinct = list(dict.fromkeys(paramValues_allOccurrences))
            paramValues_occurrencesCounts = list(map(lambda x: round(paramValues_allOccurrences.count(x)), paramValues_distinct))
            
            parameter[0].histogram.setData(list(map(str, paramValues_distinct)), paramValues_occurrencesCounts)
            parameter[1].setEnabled(True)
            parameter[1].toggle(False)
   
        #Advise future measurement 
        self._adviseStrategy.setEnabled(True)
        
        #Advice Labels
        self._adviseMeasurementLabel.setText("")
        self._adviseMeasurementPoints.setText("")
        self._adviseMeasurementStats.setText("")"""
        
            
    def calculateSunkenCost(self):
        self._sunkenCost.setText("")
        self._adviseMeasurement.setEnabled(False) 
        experiment = self.main_widget.getExperiment()
        numProc_ParamIndex = self.processesParameter.currentIndex()
        if experiment is None or numProc_ParamIndex < 0 :
            return

        #TODO: does not work anymore
        #selected_callpath = self.main_widget.getSelectedCallpath()
        selected_callpath = self.main_widget.get_selected_call_tree_nodes()
        print("DEBUG selected_callpath:",selected_callpath)

        #corehourse_struct = GPR_Interface.calculateSunkenCost(experiment, numProc_ParamIndex, selectedCallpath=selected_callpath)
        #self._sunkenCost.setText(corehourse_struct.msg)
        self._sunkenCost.setText("Dummy TEXT")

        if len(selected_callpath) < 1:
            budget = int(self._experimentBudget.value())
            if budget > 0:
                #self._sunkenCost.setText(self._sunkenCost.text() +" ("+ str(corehourse_struct.val//budget) +"%)")
                self._sunkenCost.setText(self._sunkenCost.text() +" (100%)")
                
        #if corehourse_struct.check:
        #    self._adviseMeasurement.setEnabled(True) 
        #else:
        #    self._adviseMeasurement.setEnabled(False) 
            

    def callpath_selection_changed(self):    
        #self.calculateSunkenCost()
        #TODO: fix
        pass


    def adviceMeasurment(self):
        self._adviseMeasurementLabel.setText("")
        self._adviseMeasurementPoints.setText("")
        self._adviseMeasurementStats.setText("")
        
        experiment = self.main_widget.getExperiment()
        if experiment is None:
            self.reset()
            return
        
        result_struct = GPR_Interface.adviceMeasurement(self)

        if result_struct.check:
            self._adviseMeasurementLabel.setText("Measure next at these points:")
            headerString = str(tuple([p.name for p in experiment.parameters]))
            self._adviseMeasurementPoints.setText(headerString +"\n"+ result_struct.msg)
            for parameterIndex, parameter in enumerate(self._parameters): 
                if parameterIndex >= len(experiment.parameters):
                    break

                paramValues_allOccurrences = tryCastListToInt(list(map(lambda coord: coord[parameterIndex], experiment.coordinates)))
                paramValues_distinct = list(dict.fromkeys(paramValues_allOccurrences))
                stacked = [0] * len(paramValues_distinct)
                for p in result_struct.payload:
                	stacked[paramValues_distinct.index(p[1][parameterIndex])] += 1

                parameter[0].histogram.addData(stacked)   
        else:
            self._adviseMeasurementLabel.setText(result_struct.msg)