#include "kaitai/kaitaistruct.h"
#include "proto_files/ProfilerReport.pb.h"
class nsight_cuprof_report_t;
class profile_result_t : public kaitai::kstruct {
	std::unique_ptr<ProfileResult> m_data;
public:
	profile_result_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr) :kstruct(p__io) {
		auto raw_data = p__io->read_bytes_full();
		m_data = std::make_unique<ProfileResult>();
		m_data->ParseFromString(raw_data);
	}

	ProfileResult* data() const { return m_data.get(); }
};