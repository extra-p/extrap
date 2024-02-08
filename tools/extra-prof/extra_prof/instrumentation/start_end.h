#pragma once
#include "../common_types.h"

#include "../containers/string.h"

#include "../address_mapping.h"
#include "../filesystem.h"
#include "../profile.h"
#include "commons.h"
#ifdef EXTRA_PROF_GPU
#include "gpu_instrumentation.h"
#endif

#include <numeric>
extern "C" int main(int, char**);

namespace extra_prof {

void initialize() {
    std::cerr << "EXTRA PROF: Initializing profiler" << std::endl;
    show_data_sizes();

    if (lib_enabled_features != EXTRA_PROF_ENABLED_FEATURES) {
        std::cerr << "EXTRA PROF: ERROR: Features of lib_extra_prof do not match the current executable." << std::endl;
    }

    const char* max_depth_str = std::getenv("EXTRA_PROF_MAX_DEPTH");
    if (max_depth_str != nullptr) {
        char* end;
        GLOBALS.MAX_DEPTH = std::strtoul(max_depth_str, &end, 10);
        std::cerr << "EXTRA PROF: MAX DEPTH: " << GLOBALS.MAX_DEPTH << std::endl;
    }
    auto output_dir_string = std::getenv("EXTRA_PROF_EXPERIMENT_DIRECTORY");
    if (output_dir_string == nullptr) {
        GLOBALS.output_dir = GLOBALS.name_register.defaultExperimentDirName();
    } else {
        GLOBALS.output_dir = containers::string(output_dir_string);
    }

    if (!filesystem::is_directory(GLOBALS.output_dir)) {
        filesystem::create_directory(GLOBALS.output_dir);
    }

    GLOBALS.name_register.create_address_mapping(GLOBALS.output_dir);

#ifdef EXTRA_PROF_ENERGY
    GLOBALS.cpuEnergy.start();
#endif
#ifdef EXTRA_PROF_GPU
    cupti::init();
#endif
    std::cerr << "EXTRA PROF: Profiling started" << std::endl;
}

void finalize() {
    if (!extra_prof_globals_initialised) {
        return;
    }
    auto& name_register = GLOBALS.name_register;
    std::cerr << "EXTRA PROF: Postprocessing started" << std::endl;

#ifdef EXTRA_PROF_ENERGY
    GLOBALS.cpuEnergy.stop();
#endif
#ifdef EXTRA_PROF_GPU
    cupti::finalize();
#endif

    std::cerr << "EXTRA PROF: Size of GLOBALS: " << sizeof(GLOBALS) << '\n';
    std::cerr << "EXTRA PROF: Size of calltree: "
              << GLOBALS.call_tree.calculate_size() + GLOBALS.calltree_nodes_allocator.unused_space() << '\n';
    std::cerr << "EXTRA PROF: Size of name_register: " << GLOBALS.name_register.getByteSize() << '\n';
#ifdef EXTRA_PROF_GPU
    std::cerr << "EXTRA PROF: Size of cupti_buffers: " << GLOBALS.gpu.buffer_pool.get_byte_size() << '\n';
    std::cerr << "EXTRA PROF: Size of event_stream: " << cupti::event_stream_size() << '\n';
    std::cerr << "EXTRA PROF: Size of cupti_mappings: " << cupti::cupti_mappings_size() << '\n';

#ifdef EXTRA_PROF_ENERGY
    std::cerr << "EXTRA PROF: Size of energy samples: " << GLOBALS.gpu.energySampler.getByteSize() << '\n';
#endif
#endif

    for (auto&& [thread, state] : GLOBALS.threads) {
        if (*state.isRunning) {
            char buffer[16];
            pthread_getname_np(thread, buffer, 16);
            if (strncmp("cuda", buffer, 4) != 0) {
                std::cerr << "EXTRA PROF: WARNING: Thread still running: " << buffer << ' ' << thread << '\n';
            }
        }
    }

    GLOBALS.call_tree.collectData();

    const char* slurm_procid = getenv("SLURM_PROCID");
    // if (slurm_procid == nullptr || containers::string(slurm_procid) == "0") {
    //     cupti::write_cupti_names(output_dir);
    // }

#ifdef EXTRA_PROF_EVENT_TRACE
    auto output_file_event_stream = GLOBALS.output_dir + "/event_stream";
    if (slurm_procid) {
        output_file_event_stream += slurm_procid;
    }
    output_file_event_stream += ".json";
    write_event_stream(output_file_event_stream, {{"Process", GLOBALS.cpu_event_stream}
#ifdef EXTRA_PROF_GPU
                                                  ,
                                                  {"GPU", GLOBALS.gpu.event_stream}
#endif
                                                 });
#endif

    auto output_file = GLOBALS.output_dir + "/profile";
    if (slurm_procid) {
        output_file += slurm_procid;
    }
    output_file += ".extra-prof.msgpack";
    Profile profile(GLOBALS.call_tree
#ifdef EXTRA_PROF_GPU
                    ,
                    GLOBALS.gpu.metricNames
#endif
    );
#ifdef EXTRA_PROF_ENERGY
    profile.extended_metrics.emplace_back(Metric{"energy_cpu", DeviceType::CPU});
    profile.extended_metrics.emplace_back(Metric{"energy_gpu", DeviceType::GPU});
#endif
    std::ofstream stream(output_file);
    msgpack::pack(stream, profile);

    // auto main_name = GLOBALS.name_register.check_ptr((void*)&main);

    // std::cout << "Main " << GLOBALS.call_tree.findOrAddChild(main_name->c_str())
    //           << " energy_gpu: " << GLOBALS.call_tree.findOrAddChild(main_name->c_str())->energy_gpu << '\n';
}

void finalize_on_exit() {
    auto time = get_timestamp();
    std::cerr << "EXTRA PROF: Encountered early exit. Wrapping up measurements." << std::endl;

    for (auto&& [tid, thread_state] : GLOBALS.threads) {
        while (!thread_state.timer_stack.empty()) {
            auto& current_node = thread_state.current_node;
            auto start = thread_state.timer_stack.back();
            auto duration = time - start;
            if (current_node->name() == nullptr) {
                throw std::runtime_error("EXTRA PROF: ERROR: accessing calltree root");
            }
            auto& metrics = current_node->my_metrics(true);
            metrics.duration += duration;
            metrics.visits += 1;

            // auto& metrics = current_node->my_metrics();
            // metrics.duration.fetch_add(duration, std::memory_order_acq_rel);
            // metrics.visits.fetch_add(1, std::memory_order_acq_rel);
            thread_state.timer_stack.pop_back();

#ifdef EXTRA_PROF_ENERGY
            if (GLOBALS.main_thread == tid && GLOBALS.main_thread == pthread_self()) {
                current_node->energy_cpu += GLOBALS.cpuEnergy.getEnergy() - GLOBALS.energy_stack_cpu.back();
                GLOBALS.energy_stack_cpu.pop_back();
#ifdef EXTRA_PROF_GPU
                GLOBALS.gpu.energySampler.addEntryTask(current_node, start, time);
#endif
            }
#endif

            if (current_node->parent() == nullptr) {
                throw std::runtime_error("EXTRA PROF: pop_time: Cannot go to parent of root");
            }
            current_node = current_node->parent();
#ifdef EXTRA_PROF_EVENT_TRACE
            GLOBALS.cpu_event_stream.emplace(time, EventType::END, EventEnd{thread_state.event_stack.back()});
            thread_state.event_stack.pop_back();
#endif
            thread_state.depth--;
        }
    }

    finalize();
}

} // namespace extra_prof