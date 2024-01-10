# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import logging
from typing import Mapping, Dict

from PySide6.QtGui import QColor

from extrap.entities.calltree import Node


class ModelColorMap(Mapping[Node, str]):
    colormaps = {'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
                 'extrap3': ['#8B0000', '#00008B', '#006400', '#2F4F4F', '#8B4513', '#556B2F', '#808000', '#008080',
                             '#FF00FF', '#800000', '#FF0000', '#000080', '#008000', '#00FFFF', '#800080'],
                 'grayscale': ['#000000', '#d9d9d9', '#777777'],
                 'grayscale-printable': ['#31688e', '#fde725', '#35b779', '#440154'],
                 'colorbrewer-set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628',
                                      '#f781bf', '#999999']}

    def __init__(self):
        self.color_list = []
        self.name = ''
        self.default_color = None
        self.set_colormap('default')

        self.dict_callpath_color: Dict[Node, str] = {}

    def set_colormap(self, name):
        self.name = name
        self.color_list = self.colormaps[name]
        self.default_color = self.color_list[0]

    def __getitem__(self, k):
        if k not in self.dict_callpath_color:
            logging.info(f'ModelColorMap: Color for "{k}" not found. Using fallback.')
            next_index = len(self)
            if next_index < len(self.color_list):
                self.dict_callpath_color[k] = self.color_list[next_index]
            else:
                self.dict_callpath_color[k] = self._create_color(next_index, len(self.color_list))
        else:
            return self.dict_callpath_color[k]

    def get_rgb(self, k):
        c = QColor(self.__getitem__(k))
        return c.red(), c.green(), c.blue()

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
                newcolor = self._create_color(current_index, size_of_color_list)
                self.dict_callpath_color[callpath] = newcolor
            current_index = current_index + 1

    def _create_color(self, current_index, size_of_color_list):
        offset = (current_index - size_of_color_list) % size_of_color_list
        multiple = int(current_index / size_of_color_list)
        color = self.color_list[offset]
        newcolor = QColor(color).lighter(100 + 20 * multiple).name()
        return newcolor
