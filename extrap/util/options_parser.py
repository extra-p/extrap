# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import argparse
from argparse import Action, ArgumentParser, Namespace
from typing import Sequence, Text, Optional, Union, Any

from extrap.modelers import single_parameter, multi_parameter
from extrap.modelers.modeler_options import ModelerOptionsGroup

SINGLE_PARAMETER_MODELER_KEY = '#single_parameter_modeler'
SINGLE_PARAMETER_OPTIONS_KEY = '#single_parameter_options'


def _create_parser(modeler, name=None, description=None, nested_sp=False):
    name = name or modeler.NAME
    if nested_sp:
        prog = '--options #spo'
    else:
        prog = f'--modeler {name} --options'
    sub_parser = argparse.ArgumentParser(prog=prog,
                                         description=description,
                                         prefix_chars='-_#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                         add_help=False)
    _add_options_to_parser(modeler, sub_parser)
    return sub_parser


def _modeler_option_bool(o):
    if isinstance(o, str):
        o = o.strip().lower()
        if o in ['false', 'off', 'no', '0']:
            return False
    return bool(o)


def _add_options_to_parser(modeler, parser):
    if hasattr(modeler, 'OPTIONS'):
        for name, option in modeler.OPTIONS.items():
            if isinstance(option, ModelerOptionsGroup):
                group = parser.add_argument_group(title=name)
                for o in option.options:
                    metavar = o.range or o.type.__name__.upper()
                    o_type = _modeler_option_bool if o.type is bool else o.type
                    group.add_argument(o.field, dest=o.field, action="store", type=o_type, metavar=metavar,
                                       choices=o.range,
                                       help=o.description)
            else:
                metavar = option.range or option.type.__name__.upper()
                o_type = _modeler_option_bool if option.type is bool else option.type
                parser.add_argument(name, dest=option.field, action="store", type=o_type, metavar=metavar,
                                    choices=option.range,
                                    help=option.description)


def _add_single_parameter_options(parser):
    parser.add_argument('#spm', 'single_parameter_modeler', dest=SINGLE_PARAMETER_MODELER_KEY,
                        choices=list(single_parameter.all_modelers.keys()), default=None,
                        help="Selects the single-parameter modeler used during multi-parameter modeling")
    parser.add_argument('#spo', 'single_parameter_options', dest=SINGLE_PARAMETER_OPTIONS_KEY,
                        nargs='+', metavar='KEY=VALUE', default={},
                        help="Sets the options for the single-parameter modeler", action=ModelerOptionsAction)


class ModelerHelpAction(Action):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values in single_parameter.all_modelers:
            modeler = single_parameter.all_modelers[values]
            sub_parser = _create_parser(modeler, values)
            print('Single Parameter Options')
            print('------------------------')
            sub_parser.print_help()

        if values in multi_parameter.all_modelers:
            modeler = multi_parameter.all_modelers[values]
            sub_parser = _create_parser(modeler, values)
            _add_single_parameter_options(sub_parser)
            print()
            print('Multi Parameter Options')
            print('-----------------------')
            sub_parser.print_help()
        parser.exit()


class ModelerOptionsAction(Action):
    # noinspection PyShadowingBuiltins, PyUnusedLocal
    def __init__(self, option_strings: Sequence[Text], dest: Text, help, **kwargs) -> None:
        super().__init__(option_strings, dest, help=help, **kwargs)

    @staticmethod
    def _all_spo_parser():
        modeler = single_parameter.Default()
        if not hasattr(modeler, 'OPTIONS'):
            modeler.OPTIONS = {}
        for m in single_parameter.all_modelers.values():
            if not hasattr(modeler, 'OPTIONS'):
                modeler.OPTIONS.update(m.OPTIONS)
        return _create_parser(modeler, nested_sp=True)

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: Union[Text, Sequence[Any], None],
                 option_string: Optional[Text] = ...) -> None:
        parser = None
        nested_sp = hasattr(namespace, SINGLE_PARAMETER_MODELER_KEY)
        if nested_sp:
            modeler_name = getattr(namespace, SINGLE_PARAMETER_MODELER_KEY)
            if modeler_name is None:
                parser = self._all_spo_parser()
        else:
            modeler_name = namespace.modeler

        if modeler_name in single_parameter.all_modelers:
            modeler = single_parameter.all_modelers[modeler_name]
            if hasattr(modeler, 'OPTIONS'):
                parser = _create_parser(modeler(), name=modeler_name, nested_sp=nested_sp)

        if modeler_name in multi_parameter.all_modelers:
            modeler = multi_parameter.all_modelers[modeler_name]
            if parser is None:
                parser = _create_parser(modeler(), name=modeler_name, nested_sp=nested_sp)
            _add_single_parameter_options(parser)
            _add_options_to_parser(modeler, parser)

        if parser is not None:
            options = parser.parse_args(values)
            setattr(namespace, self.dest, dict(options.__dict__))
