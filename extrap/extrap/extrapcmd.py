# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
import logging
import os
import sys
import warnings
from itertools import chain

import extrap
from extrap.entities.measurement import Measure
from extrap.entities.scaling_type import ScalingType
from extrap.fileio import experiment_io
from extrap.fileio.experiment_io import ExperimentReader
from extrap.fileio.file_reader import all_readers
from extrap.fileio.file_reader.abstract_directory_reader import AbstractScalingConversionReader
from extrap.fileio.file_reader.file_reader_mixin import TKeepValuesReader
from extrap.fileio.io_helper import save_output
from extrap.fileio.output import format_output
from extrap.modelers import multi_parameter
from extrap.modelers import single_parameter
from extrap.modelers.abstract_modeler import MultiParameterModeler
from extrap.modelers.model_generator import ModelGenerator
from extrap.util.exceptions import RecoverableError
from extrap.util.options_parser import ModelerOptionsAction, ModelerHelpAction
from extrap.util.options_parser import SINGLE_PARAMETER_MODELER_KEY, SINGLE_PARAMETER_OPTIONS_KEY
from extrap.util.progress_bar import ProgressBar


def main(args=None, prog=None):
    # argparse
    modelers_list = list(set(k.lower() for k in
                             chain(single_parameter.all_modelers.keys(), multi_parameter.all_modelers.keys())))
    parser = argparse.ArgumentParser(prog=prog, description=extrap.__description__, add_help=False)
    positional_arguments = parser.add_argument_group("Positional arguments")
    basic_arguments = parser.add_argument_group("Optional arguments")
    basic_arguments.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                                 help='Show this help message and exit')

    basic_arguments.add_argument("--version", action="version", version=extrap.__title__ + " " + extrap.__version__,
                                 help="Show program's version number and exit")
    basic_arguments.add_argument("--log", action="store", dest="log_level", type=str.lower, default='warning',
                                 choices=['debug', 'info', 'warning', 'error', 'critical'],
                                 help="Set program's log level (default: %(default)s)")

    input_options = parser.add_argument_group("Input options")
    group = input_options.add_mutually_exclusive_group(required=True)
    for reader in all_readers.values():
        group.add_argument(reader.CMD_ARGUMENT, action="store_true", default=False, dest=reader.NAME,
                           help=reader.DESCRIPTION)
    group.add_argument(ExperimentReader.CMD_ARGUMENT, action="store_true", default=False, dest=ExperimentReader.NAME,
                       help='Load Extra-P experiment and generate new models')

    names_of_scaling_conversion_readers = ", ".join(reader.NAME + " files" for reader in all_readers.values()
                                                    if issubclass(reader, AbstractScalingConversionReader))

    input_options.add_argument("--scaling", action="store", dest="scaling_type", default=ScalingType.WEAK,
                               type=ScalingType, choices=ScalingType,
                               help="Set scaling type when loading data from per-thread/per-rank files (" +
                                    names_of_scaling_conversion_readers + ") (default: %(default)s)")
    input_options.add_argument("--keep-values", action="store_true", default=False, dest='keep_values',
                               help="Keeps the original values after import")

    modeling_options = parser.add_argument_group("Modeling options")
    measure_group = modeling_options.add_mutually_exclusive_group(required=False)
    measure_group.add_argument("--median", action="store_true", dest="median",
                               help="Use median values for computation instead of mean values "
                                    "(deprecated: use --measure)")
    measure_group.add_argument("--measure", action="store", dest="measure",
                               default=Measure.MEAN.name.lower(),
                               choices=[m.name.lower() for m in Measure.choices()],
                               help="Select measure of values used for computation")
    modeling_options.add_argument("--modeler", action="store", dest="modeler", default='default', type=str.lower,
                                  choices=modelers_list,
                                  help="Selects the modeler for generating the performance models "
                                       "(default: %(default)s)")
    modeling_options.add_argument("--options", dest="modeler_options", default={}, nargs='+', metavar="KEY=VALUE",
                                  action=ModelerOptionsAction,
                                  help="Options for the selected modeler")
    modeling_options.add_argument("--help-modeler", choices=modelers_list, type=str.lower,
                                  help="Show help for modeler options and exit",
                                  action=ModelerHelpAction)

    output_options = parser.add_argument_group("Output options")
    output_options.add_argument("--out", action="store", metavar="OUTPUT_PATH", dest="out",
                                help="Specify the output path for Extra-P results")
    output_options.add_argument("--print", action="store", dest="print_type", default="all",
                                metavar='{all,all-python,all-latex,callpaths,metrics,parameters,functions,'
                                        'functions-python,functions-latex,FORMAT_STRING}',
                                help="Set which information should be displayed after modeling. Use one of "
                                     "the presets or specify a formatting string using placeholders "
                                     f"(see {extrap.__documentation_link__}/output-formatting.md). "
                                     f"(default: %(default)s)")
    output_options.add_argument("--save-experiment", action="store", metavar="EXPERIMENT_PATH", dest="save_experiment",
                                help="Saves the experiment including all models as Extra-P experiment "
                                     "(if no extension is specified, '.extra-p' is appended)")
    output_options.add_argument("--model-set-name", action="store", metavar="NAME", default='New model',
                                dest="model_name", type=str,
                                help="Sets the name of the generated set of models when outputting an experiment "
                                     '(default: "%(default)s")')
    output_options.add_argument("--disable-progress", action="store_true", dest="disable_progress", default=False,
                                help="Disables the progress bar outputs of Extra-P to stdout and stderr")

    positional_arguments.add_argument("path", metavar="FILEPATH", type=str, action="store",
                                      help="Specify a file path for Extra-P to work with")
    arguments = parser.parse_args(args)

    # set log level
    loglevel = logging.getLevelName(arguments.log_level.upper())
    # set output print type
    printtype = arguments.print_type

    # set log format location etc.
    if loglevel == logging.DEBUG:
        # import warnings
        # warnings.simplefilter('always', DeprecationWarning)
        # check if log file exists and create it if necessary
        # if not os.path.exists("../temp/extrap.log"):
        #    log_file = open("../temp/extrap.log","w")
        #    log_file.close()
        # logging.basicConfig(format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s():
        # %(message)s", level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p", filename="../temp/extrap.log", filemode="w")
        logging.basicConfig(
            format="%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)10s(): %(message)s",
            level=loglevel, datefmt="%m/%d/%Y %I:%M:%S %p")
    else:
        logging.basicConfig(
            format="%(levelname)s: %(message)s", level=loglevel)

    # check scaling type
    scaling_type = arguments.scaling_type

    # set use mean or median for computation
    if arguments.median:
        use_measure = arguments.median
    else:
        use_measure = Measure.from_str(arguments.measure)

    # save modeler output to file?
    print_path = None
    if arguments.out is not None:
        print_output = True
        print_path = arguments.out
    else:
        print_output = False

    if arguments.path is not None:
        with ProgressBar(desc='Loading file', disable=arguments.disable_progress) as pbar:
            for reader in chain(all_readers.values(), [ExperimentReader]):
                if getattr(arguments, reader.NAME):
                    file_reader = reader()

                    if issubclass(reader, AbstractScalingConversionReader):
                        file_reader.scaling_type = arguments.scaling_type
                    elif arguments.scaling_type != ScalingType.WEAK:
                        warnings.warn(
                            f"Scaling type {arguments.scaling_type} is not supported by the {reader.NAME} reader.")

                    if isinstance(file_reader, TKeepValuesReader):
                        file_reader.keep_values = arguments.keep_values
                    elif arguments.keep_values:
                        warnings.warn(
                            f"Keeping values is not supported by the {reader.NAME} reader.")

                    if reader.LOADS_FROM_DIRECTORY:
                        if os.path.isdir(arguments.path):
                            experiment = file_reader.read_experiment(arguments.path, pbar)
                        else:
                            logging.error("The given path is not valid. It must point to a directory.")
                            sys.exit(1)
                    elif os.path.isfile(arguments.path):
                        experiment = file_reader.read_experiment(arguments.path, pbar)
                    else:
                        logging.error("The given file path is not valid.")
                        sys.exit(1)

        experiment.debug()

        # initialize model generator
        model_generator = ModelGenerator(
            experiment, modeler=arguments.modeler, name=arguments.model_name, use_measure=use_measure)

        # apply modeler options
        modeler = model_generator.modeler
        if isinstance(modeler, MultiParameterModeler) and arguments.modeler_options:
            # set single-parameter modeler of multi-parameter modeler
            single_modeler = arguments.modeler_options[SINGLE_PARAMETER_MODELER_KEY]
            if single_modeler is not None:
                modeler.single_parameter_modeler = single_parameter.all_modelers[single_modeler]()
            # apply options of single-parameter modeler
            if modeler.single_parameter_modeler is not None:
                for name, value in arguments.modeler_options[SINGLE_PARAMETER_OPTIONS_KEY].items():
                    if value is not None:
                        setattr(modeler.single_parameter_modeler, name, value)

        for name, value in arguments.modeler_options.items():
            if value is not None:
                setattr(modeler, name, value)

        with ProgressBar(desc='Generating models', disable=arguments.disable_progress) as pbar:
            # create models from data
            model_generator.model_all(pbar)

        if arguments.save_experiment:
            try:
                with ProgressBar(desc='Saving experiment', disable=arguments.disable_progress) as pbar:
                    if not os.path.splitext(arguments.save_experiment)[1]:
                        arguments.save_experiment += '.extra-p'
                    experiment_io.write_experiment(experiment, arguments.save_experiment, pbar)
            except RecoverableError as err:
                logging.error('Saving experiment: ' + str(err))
                sys.exit(1)

        # format modeler output into text
        text = format_output(experiment, printtype)

        # print formatted output to command line
        print(text)

        # save formatted output to text file
        if print_output:
            save_output(text, print_path)

    else:
        logging.error("No file path given to load files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
