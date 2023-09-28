# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from abc import abstractmethod

from PySide6.QtWidgets import QWidget


class AbstractPlotWidget(QWidget):

    @abstractmethod
    def setMax(self, axis, maxValue):
        """
        This function sets the highest value that is shown on a specific axis.
        """

    @abstractmethod
    def getMax(self, axis):
        """
        This function returns the highest value that is shown on a specific axis.
        """

    @abstractmethod
    def set_initial_value(self):
        """
          This function sets the initial value for different parameters required for graph.
        """

    @abstractmethod
    def drawGraph(self):
        """
            This function is being called by paintEvent to draw the graph
        """

    @staticmethod
    @abstractmethod
    def getNumAxis():
        """
          This function returns the number of axis. If it's a 2-parameter model, it returns 2
        """
