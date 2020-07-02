"""
This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany
 
This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


import ctypes
import json


def load_cube_interface():
    """
    This method loads the cubelib dependency and the extrap cube interface.
    Then it creates an instance of the cube interface and returns this
    instance. Using this instance the shared library functions from the
    extrap cube interface can be used.
    """
    
    # Path to the Extrap Cube Interface library
    cube_interface_lib_path = "../libs/cubeinterface/CubeInterface.so"
    
    # Path to the extrap config file
    extrap_config_path = "../config.JSON"
    
    # Path to the cubelib
    cube_lib_path = read_cube_lib_path(extrap_config_path)
    
    """
    The Cubelib needs to be loaded before the ExtraP Cube Interface
    shared library otherwise Python can not find the Cube implementation
    and a segmentation fault will ocure.
    """
    
    # load cubelib
    cube = ctypes.cdll.LoadLibrary(cube_lib_path)  # @UnusedVariable

    # load and create an instace of the extrap cube interface
    cube_interface = ctypes.cdll.LoadLibrary(cube_interface_lib_path)

    return cube_interface


def read_cube_lib_path(extrap_config_path):
    """
    Reads the currently set path to the cube4.so file from the extrap
    config.JSON file. The link must be correct and not broken (e.g. happens
    when copying cube4.so) in order to load the cubelib successfully.
    """
    
    with open(extrap_config_path, "r") as config_file:
        json_data = json.load(config_file)
        
    return json_data["cube_lib_path"]
    
    