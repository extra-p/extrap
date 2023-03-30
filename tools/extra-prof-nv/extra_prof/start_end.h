#pragma once
#include "address_mapping.h"
#include "commons.h"
#include "cupti_instrumentation.h"
#include "profile.h"
#include <numeric>

namespace extra_prof {

void finalize();

void initialize() {
    std::cerr << "EXTRA PROF: Initialising" << std::endl;
    show_data_sizes();

    const char *max_depth_str = std::getenv("EXTRA_PROF_MAX_DEPTH");
    if (max_depth_str != nullptr) {
        char *end;
        MAX_DEPTH = std::strtoul(max_depth_str, &end, 10);
        if (MAX_DEPTH >= MAX_MAX_DEPTH) {
            MAX_DEPTH = MAX_MAX_DEPTH;
        }
        std::cerr << "EXTRA PROF: MAX DEPTH: " << MAX_DEPTH << std::endl;
    }
    auto output_dir_string = std::getenv("EXTRA_PROF_EXPERIMENT_DIRECTORY");
    if (output_dir_string == nullptr) {
        output_dir = std::filesystem::path(std::string("extra_prof_") + currentDateTime());
    } else {
        output_dir = std::filesystem::path(output_dir_string);
    }

    if (!std::filesystem::is_directory(output_dir)) {
        std::filesystem::create_directory(output_dir);
    }

    create_address_mapping(output_dir);
    cupti::init();
    std::atexit(finalize);
    std::cerr << "EXTRA PROF: Profiling started" << std::endl;
}

void finalize() {
    std::cerr << "EXTRA PROF: Postprocessing started" << std::endl;
    cupti::finalize();

    std::cerr << "EXTRA PROF: Size of calltree: "
              << call_tree.calculate_size() + calltree_nodes_allocator.unused_space() << '\n';
    std::cerr << "EXTRA PROF: Size of cupti_buffers: " << cupti::buffer_pool.num_buffers() * cupti::buffer_pool.size()
              << '\n';
    std::cerr << "EXTRA PROF: Size of event_stream: " << cupti::event_stream_size() << '\n';

    std::cerr << "EXTRA PROF: Size of name_register: "
              << name_register.size() * (sizeof(intptr_t) + sizeof(std::string)) +
                     std::accumulate(name_register.begin(), name_register.end(), 0,
                                     [](size_t size, auto &kv) { return size + kv.second.size(); })
              << '\n';
    std::cerr << "EXTRA PROF: Size of cupti_mappings: " << cupti::cupti_mappings_size() << '\n';

    const char *slurm_procid = getenv("SLURM_PROCID");
    // if (slurm_procid == nullptr || std::string(slurm_procid) == "0") {
    //     cupti::write_cupti_names(output_dir);
    // }

#ifdef EXTRA_PROF_EVENT_TRACE
    auto output_file_event_stream = output_dir / "event_stream";
    if (slurm_procid) {
        output_file_event_stream += slurm_procid;
    }
    output_file_event_stream += ".json";
    write_event_stream(output_file_event_stream, {cpu_event_stream, cupti::event_stream});
#endif

    auto output_file = output_dir / "profile";
    if (slurm_procid) {
        output_file += slurm_procid;
    }
    output_file += ".extra-prof.msgpack";
    std::ofstream stream(output_file);
    msgpack::pack(stream, Profile(call_tree));
}
} // namespace extra_prof