#pragma once
#include "../common_types.h"

#include "../address_mapping.h"
#include "../calltree_node.h"

#include "../events.h"
#include "../globals.h"
#include "../profile.h"

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

template <typename T>
EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(T* fn_ptr, CallTreeNodeType type = CallTreeNodeType::NONE) {
    auto& name_register = GLOBALS.name_register;
    auto* name_ptr = name_register.get_name_ptr(fn_ptr).c_str();
    return push_time(name_ptr, type);
}

template <>
EP_INLINE std::tuple<time_point, CallTreeNode*> push_time(char const* name, CallTreeNodeType type) {
    time_point time = get_timestamp();
    auto& current_node = GLOBALS.my_thread_state().current_node;
    current_node = current_node->findOrAddChild(name, type);
    GLOBALS.my_thread_state().timer_stack.push_back(time);
#ifdef EXTRA_PROF_EVENT_TRACE
    auto& ref = GLOBALS.cpu_event_stream.emplace(time, EventType::START, EventStart{current_node, pthread_self()},
                                                 pthread_self());
    GLOBALS.my_thread_state().event_stack.push_back(&ref);
#endif
    return {time, current_node};
}

template <typename T>
EP_INLINE time_point pop_time(T* fn_ptr) {
    auto& name_register = GLOBALS.name_register;
    return pop_time(name_register.check_ptr(fn_ptr)->c_str());
}
template <>
EP_INLINE time_point pop_time(char const* name) {
    time_point time = get_timestamp();
    auto& thread_state = GLOBALS.my_thread_state();
    auto& current_node = thread_state.current_node;
    auto duration = time - thread_state.timer_stack.back();
    if (current_node->name() == nullptr) {
        throw std::runtime_error("EXTRA PROF: ERROR: accessing calltree root");
    }
    if (current_node->name() != name) {
        throw std::runtime_error("EXTRA PROF: ERROR: popping different node than previously pushed");
    }
    auto& metrics = current_node->my_metrics();
    metrics.visits++;
    metrics.duration += duration;
    thread_state.timer_stack.pop_back();
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
    if (!current_node->validateChildren()) {
        std::cerr << "EXTRA PROF: ERROR: Children to big!!!" << std::endl;
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

void show_data_sizes() {
    std::cerr << "EXTRA PROF: GlobalState size: " << sizeof(GlobalState) << '\n';
    std::cerr << "EXTRA PROF: Event size: " << sizeof(Event) << '\n';
    std::cerr << "EXTRA PROF: CallTreeNode size: " << sizeof(CallTreeNode) << '\n';
}

} // namespace extra_prof