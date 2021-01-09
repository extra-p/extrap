from extrap.fileio.ncu_reader.binary_parser.nsight_cuprof_report import NsightCuprofReport
from kaitaistruct import BytesIO, KaitaiStream
from memory_profiler import profile


class NcuReport:

    def __init__(self, name):
        self.report_data = NsightCuprofReport.from_file(name)
        self.parse(self.report_data)

    def parse(self, report_data):
        self.string_table = []
        for block in report_data.blocks:
            if block.header.data.StringTable.Strings:
                self.string_table.extend(block.header.data.StringTable.Strings)
