#pragma once
#include "../common_types.h"

#include "../address_mapping.h"
#include "../calltree_node.h"

#include "../events.h"
#include "../globals.h"
#include "../profile.h"
#include "energy.h"

#ifdef EXTRA_PROF_GPU
#include "gpu_instrumentation.h"
#else
#include <time.h>
#endif

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace extra_prof {

EP_INLINE time_point get_timestamp() {
#ifdef EXTRA_PROF_GPU
    time_point time;
    CUPTI_CALL(cuptiGetTimestamp(&time));
    return time;
#else
    struct timespec time;
    int status = clock_gettime(CLOCK_REALTIME, &time);
    if (status == EINVAL || status == EFAULT || status == ENOTSUP) {
        throw std::system_error(status, std::generic_category());
    };
    time_point result = time.tv_nsec;
    result += 1000000000 * time.tv_sec;
    return result;
#endif
}

// template <typename T>
// EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(T* fn_ptr, CallTreeNodeType type = CallTreeNodeType::NONE)
// {
//     auto& name_register = GLOBALS.name_register;
//     auto* name_ptr = name_register.get_name_ptr(fn_ptr).c_str();
//     return push_time(name_ptr, type);
// }

EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(RegionType region_type, RegionID region,
                                                          CallTreeNodeType type = CallTreeNodeType::NONE,
                                                          ThreadState& state = GLOBALS.my_thread_state()) {
    time_point time = get_timestamp();
    auto& current_node = state.current_node;
    current_node = current_node->findOrAddChild(region_type, region, type);

    state.timer_stack.push_back(time);

#ifdef EXTRA_PROF_ENERGY
    if (GLOBALS.main_thread == pthread_self()) {
        GLOBALS.energy_stack_cpu.push_back(GLOBALS.cpuEnergy.getEnergy());
        // std::cout << "Current node: " << current_node << ' ' << name << '\n';
    }
#endif

#ifdef EXTRA_PROF_EVENT_TRACE
    auto& ref = GLOBALS.cpu_event_stream.emplace(time, EventType::START, EventStart{current_node, pthread_self()},
                                                 pthread_self());
    state.event_stack.push_back(&ref);
#endif
    return {time, current_node};
}

EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(const char* name,
                                                          CallTreeNodeType type = CallTreeNodeType::NONE) {
    return push_time(RegionType::NAMED_REGION, toRegionID(name), type);
}
EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(void* function_ptr,
                                                          CallTreeNodeType type = CallTreeNodeType::NONE) {
    return push_time(RegionType::FUNCTION_PTR_REGION, toRegionID(function_ptr), type);
}

EP_INLINE time_point pop_time(RegionType region_type, RegionID region,
                              ThreadState& thread_state = GLOBALS.my_thread_state()) {
    time_point time = get_timestamp();
    auto& current_node = thread_state.current_node; // Must be a reference, so that we can change it below
    auto start = thread_state.timer_stack.back();
    auto duration = time - start;
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
    if (current_node->is_root()) {
        throw std::runtime_error("EXTRA PROF: ERROR: accessing calltree root");
    }
    if (current_node->region_type != region_type && current_node->region.function_ptr != region.function_ptr) {
        throw std::runtime_error("EXTRA PROF: ERROR: popping different node than previously pushed");
    }
#endif
    auto& metrics = current_node->my_metrics();
    metrics.duration += duration;
    metrics.visits += 1;
    // auto& metrics = current_node->my_metrics();
    // metrics.visits.fetch_add(1, std::memory_order_acq_rel);
    // metrics.duration.fetch_add(duration, std::memory_order_acq_rel);
    thread_state.timer_stack.pop_back();
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
    if (!current_node->validateChildren()) {
        std::cerr << "EXTRA PROF: ERROR: Children to big!!!" << std::endl;
    }
#endif

#ifdef EXTRA_PROF_ENERGY
    if (GLOBALS.main_thread == pthread_self()) {
        current_node->energy_cpu.fetch_add(GLOBALS.cpuEnergy.getEnergy() - GLOBALS.energy_stack_cpu.back(),
                                           std::memory_order_acq_rel);
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
    return time;
}
EP_INLINE time_point pop_time(const char* name) { return pop_time(RegionType::NAMED_REGION, toRegionID(name)); }

void show_data_sizes() {
    std::cerr << "EXTRA PROF: GlobalState size: " << sizeof(GlobalState) << '\n';
    std::cerr << "EXTRA PROF: Event size: " << sizeof(Event) << '\n';
    std::cerr << "EXTRA PROF: CallTreeNode size: " << sizeof(CallTreeNode) << '\n';
}

} // namespace extra_prof