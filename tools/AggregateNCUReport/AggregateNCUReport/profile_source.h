#include "kaitai/kaitaistruct.h"
#include "proto_files/ProfilerResults.pb.h"
class nsight_cuprof_report_t;

class profile_source_t : public kaitai::kstruct {
	std::unique_ptr<ProfilerSourceMessage> m_data;
public:
	profile_source_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr) :kstruct(p__io) {
		auto raw_data = p__io->read_bytes_full();
		//m_data = std::make_unique<ProfilerSourceMessage>();
		//m_data->ParseFromString(raw_data);
	}

	ProfilerSourceMessage* data() const { return m_data.get(); }
};