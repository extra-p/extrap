# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2022, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import importlib
import sys
from types import ModuleType

from extrap.util.deprecation import deprecated
from extrap.util.progress_bar import DUMMY_PROGRESS


def _make_compat_module(module_name, method_name, reader_class_name, method_impl=None):
    if not method_impl:
        @deprecated(f"Use extrap.fileio.file_reader.{module_name}.{reader_class_name}.read_experiment instead.",
                    f"extrap.fileio.{module_name}.{method_name} is deprecated.")
        def read_file_method(path, pbar=DUMMY_PROGRESS):
            reader_class = getattr(importlib.import_module('extrap.fileio.file_reader.' + module_name),
                                   reader_class_name)
            return reader_class().read_experiment(path, pbar)
    else:
        read_file_method = method_impl

    module = ModuleType('extrap.fileio.' + module_name)
    setattr(module, method_name, read_file_method)
    sys.modules['extrap.fileio.' + module_name] = module


_make_compat_module('text_file_reader', 'read_text_file', 'TextFileReader')
_make_compat_module('talpas_file_reader', 'read_talpas_file', 'TalpasFileReader')
_make_compat_module('json_file_reader', 'read_json_file', 'JsonFileReader')
_make_compat_module('extrap3_experiment_reader', 'read_extrap3_experiment', 'Extrap3ExperimentFileReader')


@deprecated("Use extrap.fileio.file_reader.cube_file_reader2.CubeFileReader2.read_experiment instead.",
            "extrap.fileio.cube_file_reader2.read_cube_file is deprecated.")
def _read_cube_file(dir_name, scaling_type, pbar=DUMMY_PROGRESS, selected_metrics=None):
    from extrap.fileio.file_reader.cube_file_reader2 import CubeFileReader2
    return CubeFileReader2().read_cube_file(dir_name, scaling_type, pbar, selected_metrics)


@deprecated("Use extrap.fileio.file_reader.jsonlines_file_reader.read_jsonlines_file instead.",
            "extrap.fileio.jsonlines_file_reader.read_jsonlines_file is deprecated.")
def _read_jsonlines_file(path, progress_bar=DUMMY_PROGRESS):
    from extrap.fileio.file_reader import jsonlines_file_reader
    return jsonlines_file_reader.read_jsonlines_file(path, progress_bar)


_make_compat_module('cube_file_reader2', 'read_cube_file', 'CubeFileReader2', _read_cube_file)
_make_compat_module('jsonlines_file_reader', 'read_jsonlines_file', None, _read_jsonlines_file)
