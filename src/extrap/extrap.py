"""
This file is part of the Extra-P software (https://github.com/MeaParvitas/Extra-P)

Copyright (c) 2020,
Technische Universitaet Darmstadt, Germany

This software may be modified and distributed under the terms of
a BSD-style license. See the LICENSE file in the base
directory for details.
"""


import logging
import argparse
import os
import sys

from itertools import chain

# from experiment import Experiment
# from fileio.fileio import open_json
from fileio.cube_file_reader2 import read_cube_file
from fileio.text_file_reader import read_text_file
from fileio.talpas_file_reader import read_talpas_file
from fileio.json_file_reader import read_json_file
from fileio.io_helper import save_output
from fileio.io_helper import format_output
from modelers.modelgenerator import ModelGenerator
from modelers import single_parameter
from modelers import multi_parameter


def Main():

    # Initialize()

    # argparse
    programname = "Extra-P"
    modelers_list = ", ".join(
        chain(single_parameter.all_modelers.keys(), multi_parameter.all_modelers.keys()))
    parser = argparse.ArgumentParser(description=programname)

    parser.add_argument("--log", action="store", dest="log_level", help="set program's log level [INFO (default), DEBUG]")
    parser.add_argument("--version", action="version", version=programname + " 4.0")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cube", action="store_true", default=False, dest="cube", help="load data from cube files")
    group.add_argument("--text", action="store_true", default=False, dest="text", help="load data from text files")
    group.add_argument("--talpas", action="store_true", default=False, dest="talpas", help="load data from talpas data format")
    group.add_argument("--json", action="store_true", default=False, dest="json", help="load data from json file")

    parser.add_argument("--modeler", action="store", dest="modeler", default='default',
                        help="select one of the following modelers: "+modelers_list)

    parser.add_argument("--scaling", action="store", dest="scaling_type", help="set weak or strong scaling when loading data from cube files [WEAK (default), STRONG]")
    parser.add_argument("--median", action="store_true", dest="median", help="use median values for computation instead of mean values")
    parser.add_argument("--out", action="store", dest="out", help="specify the output path for Extra-P results")
    parser.add_argument("--print", action="store", dest="print_type", help="set which information should be displayed after modeling [ALL (default), CALLPATHS, METRICS, PARAMETERS, FUNCTIONS]")

    parser.add_argument("path", metavar="FILEPATH", action="store", help="specify a file path for Extra-P to work with")

    arguments = parser.parse_args()

    # set log level
    if not arguments.log_level is None:
        loglevel = arguments.log_level.upper()
        if loglevel == "DEBUG":
            loglevel = logging.DEBUG
        elif loglevel == "INFO":
            loglevel = logging.INFO
        else:
            loglevel = logging.INFO
    else:
        loglevel = logging.INFO

    # set output print type
    printtype = "ALL"
    if not arguments.print_type is None:
        printtype = arguments.print_type.upper()
        if printtype == "ALL":
            printtype = "ALL"
        elif printtype == "CALLPATHS":
            printtype = "CALLPATHS"
        elif printtype == "METRICS":
            printtype = "METRICS"
        elif printtype == "PARAMETERS":
            printtype = "PARAMETERS"
        elif printtype == "FUNCTIONS":
            printtype = "FUNCTIONS"
        else:
            printtype = "ALL"
            logging.warning("Invalid print type.")

    # set log format location etc.
    if loglevel == logging.DEBUG:
        import warnings
        warnings.simplefilter('always', DeprecationWarning)
        # check if log file exists and create it if necessary
        # if not os.path.exists("../temp/extrap.log"):
        #    log_file = open("../temp/extrap.log","w")
        #    log_file.close()
        # logging.basicConfig(format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s", level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p", filename="../temp/extrap.log", filemode="w")
        logging.basicConfig(format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
                            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    # check scaling type
    scaling_type = "weak"
    if not arguments.scaling_type is None:
        if arguments.scaling_type.lower() == "weak":
            scaling_type = "weak"
        elif arguments.scaling_type.lower() == "strong":
            scaling_type = "strong"
        else:
            scaling_type = "weak"
            logging.warning("Invalid scaling type. Supported types are WEAK (default) and STRONG. Using weak scaling instead.")

    # use mean or median?
    use_median = arguments.median

    # save modeler output to file?
    print_output = False
    print_path = None
    if not arguments.out is None:
        print_output = True
        print_path = arguments.out
    else:
        print_output = False

    if not arguments.path is None:
        if arguments.cube == True:
            # load data from cube files
            if os.path.isdir(arguments.path):
                experiment = read_cube_file(arguments.path, scaling_type)
            else:
                logging.error("The given file path is not valid.")
                return 1
        elif os.path.isfile(arguments.path):
            if arguments.text == True:
                # load data from text files
                experiment = read_text_file(arguments.path)
            elif arguments.talpas == True:
                # load data from talpas format
                experiment = read_talpas_file(arguments.path)
            elif arguments.json == True:
                # load data from json file
                experiment = read_json_file(arguments.path)
            else:
                logging.error(
                    "The file format specifier is missing.")
                return 1
        else:
            logging.error("The given file path is not valid.")
            return 1

        # TODO: debug code
        experiment.debug()

        # initialize model generator
        model_generator = ModelGenerator(
            experiment, modeler=arguments.modeler, use_median=use_median)

        # create models from data
        model_generator.model_all()

        # format modeler output into text
        text = format_output(experiment, printtype)

        # print formatted output to command line
        print(text)

        # save formatted output to text file
        if print_output == True:
            save_output(text, print_path)

    else:
        logging.error("No file path given to load files.")
        return 1


if __name__ == "__main__":
    Main()
