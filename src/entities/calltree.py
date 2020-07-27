"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


class Node:

    def __init__(self, name, path):
        self.name = name
        self.childs = []
        self.path = path

    def add_child_node(self, child_node):
        self.childs.append(child_node)

    def get_childs(self):
        return self.childs

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


class CallTree(Node):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__('', '')

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
