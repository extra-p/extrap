# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

import zipfile
from pathlib import Path
from typing import Union
from zipfile import ZipFile

from marshmallow import ValidationError

from extrap.entities.experiment import ExperimentSchema, Experiment
from extrap.fileio.file_reader import FileReader
from extrap.fileio.values_io import ValueWriter, ValueReader
from extrap.util.exceptions import FileFormatError, RecoverableError
from extrap.util.progress_bar import DUMMY_PROGRESS, ProgressBar

EXPERIMENT_DATA_FILE = 'experiment.json'


def read_experiment(path, progress_bar=DUMMY_PROGRESS) -> Experiment:
    progress_bar.total += 3
    schema = ExperimentSchema()
    schema.set_progress_bar(progress_bar)
    try:
        with ZipFile(path, 'r', allowZip64=True) as file:
            progress_bar.update()
            data = file.read(EXPERIMENT_DATA_FILE).decode("utf-8")
            progress_bar.update()
            try:
                with ValueReader(file) as value_reader:
                    schema.set_value_io(value_reader)
                    experiment = schema.loads(data)
                    progress_bar.update()
                    return experiment
            except ValidationError as v_err:
                raise FileFormatError(str(v_err)) from v_err
    except (IOError, zipfile.BadZipFile) as err:
        raise RecoverableError(str(err)) from err


class ExperimentReader(FileReader):
    NAME = 'experiment'
    CMD_ARGUMENT = '--experiment'
    GUI_ACTION = '&Open experiment'
    DESCRIPTION = 'Load Extra-P experiment'
    FILTER = 'Experiments (*.extra-p)'
    GENERATE_MODELS_AFTER_LOAD = False

    def read_experiment(self, path: Union[Path, str], progress_bar: ProgressBar = DUMMY_PROGRESS) -> Experiment:
        return read_experiment(path, progress_bar)


def write_experiment(experiment, path, progress_bar=DUMMY_PROGRESS):
    progress_bar.update(0)
    progress_bar.total += 4
    schema = ExperimentSchema()
    progress_bar.total += sum((len(m.models) for m in experiment.modelers), 0)
    schema.set_progress_bar(progress_bar)
    progress_bar.update()
    try:
        with ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as file:
            value_writer = ValueWriter(file)
            schema.set_value_io(value_writer)
            progress_bar.update()
            try:
                data = schema.dumps(experiment)
                progress_bar.update()
                value_writer.flush()
                file.writestr(EXPERIMENT_DATA_FILE, data)
                progress_bar.update()
            except ValidationError as v_err:
                raise FileFormatError(str(v_err)) from v_err
    except (IOError, FileNotFoundError, zipfile.BadZipFile) as err:
        raise RecoverableError(str(err)) from err
