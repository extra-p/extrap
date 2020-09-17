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
