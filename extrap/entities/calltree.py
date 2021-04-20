# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from typing import cast

from extrap.entities.callpath import Callpath


class Node:

    def __init__(self, name: str, path: Callpath, childs=None):
        self.name = name
        self.childs = [] if childs is None else childs
        self.path = path

    def add_child_node(self, child_node):
        self.childs.append(child_node)

    def get_childs(self):
        return self.childs

    def find_child(self, child_node_name):
        for child in self.childs:
            if child.name == child_node_name:
                return child
        return None

    def child_exists(self, child_node_name):
        for i in range(len(self.childs)):
            child_name = self.childs[i].name
            if child_name == child_node_name:
                return True
        return False

    def __repr__(self):
        if self.childs:
            return f'{self.name}->[{", ".join(c.name for c in self.childs)}]'
        else:
            return self.name

    def __iter__(self):
        return iter(self.childs)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.name == other.name and self.path == other.path and self.childs == other.childs

    def __hash__(self):
        return hash((self.name, self.path))

    def _generate_code_representation(self):
        return f'Node("{self.name}",Callpath("{self.path}"),[{",".join(c._generate_code_representation() for c in self.childs)}])'

    def exactly_equal(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        elif self is other:
            return True
        else:
            return self.name == other.name and self.path.exactly_equal(other.path) and all(
                a.exactly_equal(b) for a, b in zip(self.childs, other.childs))

class CallTree(Node):
    """
    Represents a calltree with nodes containing the corresponding callpaths
    """

    def __init__(self):
        super().__init__('', cast(Callpath, None))

    def add_node(self, node):
        self.childs.append(node)

    def get_node(self, node_name):
        for i in range(len(self.childs)):
            name = self.childs[i].name
            if name == node_name:
                return self.childs[i]

    def get_nodes(self):
        return self.childs

    def node_exist(self, name):
        for i in range(len(self.childs)):
            node = self.childs[i]
            if node.name == name:
                return True
        return False

    def print_tree(self):
        level = ""
        for i in range(len(self.childs)):
            string_name = self.childs[i].name
            print("-" + str(string_name))
            self.print_childs(self.childs[i].childs, level)

    def print_childs(self, childs, level):
        level = level + "\t"
        for i in range(len(childs)):
            child = childs[i]
            child_name = child.name
            print(level + "-" + str(child_name))
            if len(child.childs) != 0:
                self.print_childs(child.childs, level)
            else:
                # do nothing
                pass
