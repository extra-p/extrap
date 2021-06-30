# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2021, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
import warnings
from typing import Mapping, Dict

from PySide2.QtGui import QColor

from extrap.entities.calltree import Node


class ModelColorMap(Mapping[Node, str]):
    def __init__(self):
        self.color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.default_color = self.color_list[0]
        self.graph_light_color_list = []
        # ['#8B0000', '#00008B', '#006400', '#2F4F4F', '#8B4513', '#556B2F',
        #  '#808000', '#008080', '#FF00FF', '#800000', '#FF0000', '#000080', '#008000', '#00FFFF', '#800080']
        self.dict_callpath_color: Dict[Node, str] = {}

    def __getitem__(self, k):
        try:
            return self.dict_callpath_color[k]
        except KeyError:
            logging.warning("ModelColorMap: Color not found. Using fallback.")
            return '#FF00FF'

    def __len__(self):
        return len(self.dict_callpath_color)

    def __iter__(self):
        return iter(self.dict_callpath_color)

    def items(self):
        return self.dict_callpath_color.items()

    def update(self, call_tree_nodes):
        current_index = 0
        size_of_color_list = len(self.color_list)
        self.dict_callpath_color.clear()
        for callpath in call_tree_nodes:
            if current_index < size_of_color_list:
                self.dict_callpath_color[callpath] = self.color_list[current_index]
            else:
                offset = (current_index - size_of_color_list) % size_of_color_list
                multiple = int(current_index / size_of_color_list)
                color = self.color_list[offset]
                newcolor = QColor(color).lighter(100 + 20 * multiple).name()
                self.dict_callpath_color[callpath] = newcolor
            current_index = current_index + 1
