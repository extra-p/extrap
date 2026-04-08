#pragma once

// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "kaitai/kaitaistruct.h"
#include <stdint.h>
#include <memory>
#include "file_header.h"
#include "profile_result.h"
#include "profile_source.h"
#include "block_header.h"
#include <vector>

#if KAITAI_STRUCT_VERSION < 9000L
#error "Incompatible Kaitai Struct C++/STL API: version 0.9 or later is required"
#endif
class file_header_t;
class profile_result_t;
class profile_source_t;
class block_header_t;

class nsight_cuprof_report_t : public kaitai::kstruct {

public:
    class payload_result_t;
    class payload_source_t;
    class payload_entries_t;
    class block_t;

    nsight_cuprof_report_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

private:
    void _read();
    void _clean_up();

public:
    ~nsight_cuprof_report_t();


    class payload_result_t : public kaitai::kstruct {

    public:

        payload_result_t(kaitai::kstream* p__io, nsight_cuprof_report_t::payload_entries_t* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

    private:
        void _read();
        void _clean_up();

    public:
        ~payload_result_t();

    private:
        uint32_t m_sizeof_payload;
        std::unique_ptr<profile_result_t> m_entry;
        nsight_cuprof_report_t* m__root;
        nsight_cuprof_report_t::payload_entries_t* m__parent;
        std::string m__raw_entry;
        std::unique_ptr<kaitai::kstream> m__io__raw_entry;

    public:
        uint32_t sizeof_payload() const { return m_sizeof_payload; }
        profile_result_t* entry() const { return m_entry.get(); }
        nsight_cuprof_report_t* _root() const { return m__root; }
        nsight_cuprof_report_t::payload_entries_t* _parent() const { return m__parent; }
        std::string _raw_entry() const { return m__raw_entry; }
        kaitai::kstream* _io__raw_entry() const { return m__io__raw_entry.get(); }
    };

    class payload_source_t : public kaitai::kstruct {

    public:

        payload_source_t(kaitai::kstream* p__io, nsight_cuprof_report_t::payload_entries_t* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

    private:
        void _read();
        void _clean_up();

    public:
        ~payload_source_t();

    private:
        uint32_t m_sizeof_payload;
        std::unique_ptr<profile_source_t> m_entry;
        nsight_cuprof_report_t* m__root;
        nsight_cuprof_report_t::payload_entries_t* m__parent;
        std::string m__raw_entry;
        std::unique_ptr<kaitai::kstream> m__io__raw_entry;

    public:
        uint32_t sizeof_payload() const { return m_sizeof_payload; }
        profile_source_t* entry() const { return m_entry.get(); }
        nsight_cuprof_report_t* _root() const { return m__root; }
        nsight_cuprof_report_t::payload_entries_t* _parent() const { return m__parent; }
        std::string _raw_entry() const { return m__raw_entry; }
        kaitai::kstream* _io__raw_entry() const { return m__io__raw_entry.get(); }
    };

    class payload_entries_t : public kaitai::kstruct {

    public:

        payload_entries_t(uint32_t p_num_sources, uint32_t p_num_results, kaitai::kstream* p__io, nsight_cuprof_report_t::block_t* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

    private:
        void _read();
        void _clean_up();

    public:
        ~payload_entries_t();

    private:
        std::unique_ptr<std::vector<std::unique_ptr<payload_source_t>>> m_sources;
        std::unique_ptr<std::vector<std::unique_ptr<payload_result_t>>> m_results;
        uint32_t m_num_sources;
        uint32_t m_num_results;
        nsight_cuprof_report_t* m__root;
        nsight_cuprof_report_t::block_t* m__parent;

    public:
        std::vector<std::unique_ptr<payload_source_t>>* sources() const { return m_sources.get(); }
        std::vector<std::unique_ptr<payload_result_t>>* results() const { return m_results.get(); }
        uint32_t num_sources() const { return m_num_sources; }
        uint32_t num_results() const { return m_num_results; }
        nsight_cuprof_report_t* _root() const { return m__root; }
        nsight_cuprof_report_t::block_t* _parent() const { return m__parent; }
    };

    class block_t : public kaitai::kstruct {

    public:

        block_t(kaitai::kstream* p__io, nsight_cuprof_report_t* p__parent = nullptr, nsight_cuprof_report_t* p__root = nullptr);

    private:
        void _read();
        void _clean_up();

    public:
        ~block_t();

    private:
        uint32_t m_sizeof_header;
        std::unique_ptr<block_header_t> m_header;
        std::unique_ptr<payload_entries_t> m_payload;
        nsight_cuprof_report_t* m__root;
        nsight_cuprof_report_t* m__parent;
        std::string m__raw_header;
        std::unique_ptr<kaitai::kstream> m__io__raw_header;
        std::string m__raw_payload;
        std::unique_ptr<kaitai::kstream> m__io__raw_payload;

    public:
        uint32_t sizeof_header() const { return m_sizeof_header; }
        block_header_t* header() const { return m_header.get(); }
        payload_entries_t* payload() const { return m_payload.get(); }
        nsight_cuprof_report_t* _root() const { return m__root; }
        nsight_cuprof_report_t* _parent() const { return m__parent; }
        std::string _raw_header() const { return m__raw_header; }
        kaitai::kstream* _io__raw_header() const { return m__io__raw_header.get(); }
        std::string _raw_payload() const { return m__raw_payload; }
        kaitai::kstream* _io__raw_payload() const { return m__io__raw_payload.get(); }
    };

private:
    std::string m_magic;
    uint32_t m_sizeof_header;
    std::unique_ptr<file_header_t> m_header;
    std::unique_ptr<std::vector<std::unique_ptr<block_t>>> m_blocks;
    nsight_cuprof_report_t* m__root;
    kaitai::kstruct* m__parent;
    std::string m__raw_header;
    std::unique_ptr<kaitai::kstream> m__io__raw_header;

public:
    std::string magic() const { return m_magic; }
    uint32_t sizeof_header() const { return m_sizeof_header; }
    file_header_t* header() const { return m_header.get(); }
    std::vector<std::unique_ptr<block_t>>* blocks() const { return m_blocks.get(); }
    nsight_cuprof_report_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
    std::string _raw_header() const { return m__raw_header; }
    kaitai::kstream* _io__raw_header() const { return m__io__raw_header.get(); }
};
