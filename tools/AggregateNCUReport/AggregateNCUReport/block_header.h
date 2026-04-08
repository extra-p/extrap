
#include "kaitai/kaitaistruct.h"
#include "proto_files/ProfilerReport.pb.h"
class nsight_cuprof_report_t;
class i_block_header_t : public kaitai::kstruct {
public:

    i_block_header_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

private:
    void _read();
    void _clean_up();

public:
    ~i_block_header_t();

private:
    bool f_num_sources;
    uint32_t m_num_sources;

public:
    virtual uint32_t num_sources();

private:
    bool f_num_results;
    uint32_t m_num_results;

public:
    virtual uint32_t num_results();

private:
    bool f_payload_size;
    uint32_t m_payload_size;

public:
    virtual uint32_t payload_size() ;

private:
    nsight_cuprof_report_t* m__root;
    kaitai::kstruct* m__parent;

public:
    nsight_cuprof_report_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

inline i_block_header_t::i_block_header_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    f_num_sources = false;
    f_num_results = false;
    f_payload_size = false;
    _read();
}

inline void i_block_header_t::_read() {
}

inline i_block_header_t::~i_block_header_t() {
    _clean_up();
}

inline void i_block_header_t::_clean_up() {
}

inline uint32_t i_block_header_t::num_sources() {
    if (f_num_sources)
        return m_num_sources;
    m_num_sources = 0;
    f_num_sources = true;
    return m_num_sources;
}

inline uint32_t i_block_header_t::num_results() {
    if (f_num_results)
        return m_num_results;
    m_num_results = 0;
    f_num_results = true;
    return m_num_results;
}

inline uint32_t i_block_header_t::payload_size() {
    if (f_payload_size)
        return m_payload_size;
    m_payload_size = 0;
    f_payload_size = true;
    return m_payload_size;
}

class block_header_t : public i_block_header_t {
	std::unique_ptr<BlockHeader> m_data;
public:
	block_header_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr) : i_block_header_t(p__io, p__parent, p__root) {
		auto raw_data = p__io->read_bytes_full();
		m_data = std::make_unique<BlockHeader>();
		m_data->ParseFromString(raw_data);
	}

    BlockHeader* data() const { return m_data.get(); }
    std::unique_ptr<BlockHeader>&& retrieve_data_ptr() { return std::move(m_data); }

public:
    virtual uint32_t num_sources() {
        return m_data->numsources();
    }

    virtual uint32_t num_results() {
        return m_data->numresults();
    }

    virtual uint32_t payload_size() {
        return m_data->payloadsize();
    }

	~block_header_t(){}
};