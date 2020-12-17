from extrap.fileio.ncu_reader.binary_parser.nsight_cuprof_report import NsightCuprofReport
from kaitaistruct import BytesIO, KaitaiStream
from memory_profiler import profile


class NcuReport:
    @profile
    def __init__(self, name):
        self.report = NsightCuprofReport.from_file(name)


