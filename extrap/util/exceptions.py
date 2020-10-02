# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

class RecoverableError(RuntimeError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FileFormatError(RecoverableError):
    NAME = 'File Format Error'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class InvalidExperimentError(RecoverableError):
    NAME = 'Invalid Experiment Error'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CancelProcessError(RecoverableError):
    NAME = 'Canceled Process'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
