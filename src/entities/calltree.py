"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

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
    

class CallTree:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.nodes = []
    
    def add_node(self, node):
        self.nodes.append(node)
        
    def get_node(self, node_name):
        for i in range(len(self.nodes)):
            name = self.nodes[i].name
            if name == node_name:
                return self.nodes[i]
            
    def get_nodes(self):
        return self.nodes
        
    def node_exist(self, name):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.name == name:
                return True
        return False
    
    def print_tree(self):
        level = ""
        for i in range(len(self.nodes)):
            string_name = self.nodes[i].name
            print("-"+str(string_name))
            self.print_childs(self.nodes[i].childs, level)
            
    def print_childs(self, childs, level):
        level = level+"\t"
        for i in range(len(childs)):
            child = childs[i]
            child_name = child.name
            print(level+"-"+str(child_name))
            if len(child.childs) != 0:
                self.print_childs(child.childs, level)
            else:
                # do nothing
                pass
        
                    