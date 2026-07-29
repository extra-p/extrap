// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "nsight_cuprof_report.h"
#include "kaitai/exceptions.h"

nsight_cuprof_report_t::nsight_cuprof_report_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = this;
    m_header = nullptr;
    m__io__raw_header = nullptr;
    m_blocks = nullptr;
    _read();
}

void nsight_cuprof_report_t::_read() {
    m_magic = m__io->read_bytes(4);
    if (!(magic() == std::string("\x4E\x56\x52\x00", 4))) {
        throw kaitai::validation_not_equal_error<std::string>(std::string("\x4E\x56\x52\x00", 4), magic(), _io(), std::string("/seq/0"));
    }
    m_sizeof_header = m__io->read_u4le();
    m__raw_header = m__io->read_bytes(sizeof_header());
    m__io__raw_header = std::unique_ptr<kaitai::kstream>(new kaitai::kstream(m__raw_header));
    m_header = std::unique_ptr<file_header_t>(new file_header_t(m__io__raw_header.get()));
    m_blocks = std::unique_ptr<std::vector<std::unique_ptr<block_t>>>(new std::vector<std::unique_ptr<block_t>>());
    {
        int i = 0;
        while (!m__io->is_eof()) {
            m_blocks->push_back(std::move(std::unique_ptr<block_t>(new block_t(m__io, this, m__root))));
            i++;
        }
    }
}

nsight_cuprof_report_t::~nsight_cuprof_report_t() {
    _clean_up();
}

void nsight_cuprof_report_t::_clean_up() {
}



nsight_cuprof_report_t::payload_result_t::payload_result_t(kaitai::kstream* p__io, nsight_cuprof_report_t::payload_entries_t* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_entry = nullptr;
    m__io__raw_entry = nullptr;
    _read();
}

void nsight_cuprof_report_t::payload_result_t::_read() {
    m_sizeof_payload = m__io->read_u4le();
    m__raw_entry = m__io->read_bytes(sizeof_payload());
    m__io__raw_entry = std::unique_ptr<kaitai::kstream>(new kaitai::kstream(m__raw_entry));
    m_entry = std::unique_ptr<profile_result_t>(new profile_result_t(m__io__raw_entry.get()));
}

nsight_cuprof_report_t::payload_result_t::~payload_result_t() {
    _clean_up();
}

void nsight_cuprof_report_t::payload_result_t::_clean_up() {
}

nsight_cuprof_report_t::payload_source_t::payload_source_t(kaitai::kstream* p__io, nsight_cuprof_report_t::payload_entries_t* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_entry = nullptr;
    m__io__raw_entry = nullptr;
    _read();
}

void nsight_cuprof_report_t::payload_source_t::_read() {
    m_sizeof_payload = m__io->read_u4le();
    m__raw_entry = m__io->read_bytes(sizeof_payload());
    m__io__raw_entry = std::unique_ptr<kaitai::kstream>(new kaitai::kstream(m__raw_entry));
    m_entry = std::unique_ptr<profile_source_t>(new profile_source_t(m__io__raw_entry.get()));
}

nsight_cuprof_report_t::payload_source_t::~payload_source_t() {
    _clean_up();
}

void nsight_cuprof_report_t::payload_source_t::_clean_up() {
}

nsight_cuprof_report_t::payload_entries_t::payload_entries_t(uint32_t p_num_sources, uint32_t p_num_results, kaitai::kstream* p__io, nsight_cuprof_report_t::block_t* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_num_sources = p_num_sources;
    m_num_results = p_num_results;
    m_sources = nullptr;
    m_results = nullptr;
    _read();
}

void nsight_cuprof_report_t::payload_entries_t::_read() {
    int l_sources = num_sources();
    m_sources = std::unique_ptr<std::vector<std::unique_ptr<payload_source_t>>>(new std::vector<std::unique_ptr<payload_source_t>>());
    m_sources->reserve(l_sources);
    for (int i = 0; i < l_sources; i++) {
        m_sources->push_back(std::move(std::unique_ptr<payload_source_t>(new payload_source_t(m__io, this, m__root))));
    }
    int l_results = num_results();
    m_results = std::unique_ptr<std::vector<std::unique_ptr<payload_result_t>>>(new std::vector<std::unique_ptr<payload_result_t>>());
    m_results->reserve(l_results);
    for (int i = 0; i < l_results; i++) {
        m_results->push_back(std::move(std::unique_ptr<payload_result_t>(new payload_result_t(m__io, this, m__root))));
    }
}

nsight_cuprof_report_t::payload_entries_t::~payload_entries_t() {
    _clean_up();
}

void nsight_cuprof_report_t::payload_entries_t::_clean_up() {
}

nsight_cuprof_report_t::block_t::block_t(kaitai::kstream* p__io, nsight_cuprof_report_t* p__parent, nsight_cuprof_report_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = p__root;
    m_header = nullptr;
    m__io__raw_header = nullptr;
    m_payload = nullptr;
    m__io__raw_payload = nullptr;
    _read();
}

void nsight_cuprof_report_t::block_t::_read() {
    m_sizeof_header = m__io->read_u4le();
    m__raw_header = m__io->read_bytes(sizeof_header());
    m__io__raw_header = std::unique_ptr<kaitai::kstream>(new kaitai::kstream(m__raw_header));
    m_header = std::unique_ptr<block_header_t>(new block_header_t(m__io__raw_header.get()));
    m__raw_payload = m__io->read_bytes(static_cast<i_block_header_t*>(header())->payload_size());
    m__io__raw_payload = std::unique_ptr<kaitai::kstream>(new kaitai::kstream(m__raw_payload));
    m_payload = std::unique_ptr<payload_entries_t>(new payload_entries_t(static_cast<i_block_header_t*>(header())->num_sources(), static_cast<i_block_header_t*>(header())->num_results(), m__io__raw_payload.get(), this, m__root));
}

nsight_cuprof_report_t::block_t::~block_t() {
    _clean_up();
}

void nsight_cuprof_report_t::block_t::_clean_up() {
}
