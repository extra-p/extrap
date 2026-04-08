#include "kaitai/kaitaistruct.h"
#include "proto_files/ProfilerReport.pb.h"
class nsight_cuprof_report_t;
class file_header_t : public kaitai::kstruct {
	std::unique_ptr<FileHeader> m_data;
public:
	file_header_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr) :kstruct(p__io) {
		auto raw_data = p__io->read_bytes_full();
		m_data = std::make_unique<FileHeader>();
		m_data->ParseFromString(raw_data);
	}

	FileHeader* data() const { return m_data.get(); }
};