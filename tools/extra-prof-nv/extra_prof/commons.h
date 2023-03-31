
#pragma once
#include "address_mapping.h"
#include "calltree_node.h"
#include "common_types.h"
#include "cupti_instrumentation.h"
#include "events.h"
#include "globals.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace extra_prof {

template <typename T>
inline std::tuple<time_point, CallTreeNode *> push_time(T *fn_ptr, CallTreeNodeType type = CallTreeNodeType::NONE) {
    auto &name_register = GLOBALS.name_register;
    auto &adress_offset = GLOBALS.adress_offset;
    auto ptr = reinterpret_cast<intptr_t>(fn_ptr);
    if (name_register.find(ptr - adress_offset) == name_register.end()) {
        Dl_info info;
        int status = dladdr(fn_ptr, &info);
        if (status != 0) {
            intptr_t base_ptr = reinterpret_cast<intptr_t>(info.dli_fbase);
            if (name_register.find(ptr - base_ptr) == name_register.end()) {
                if (info.dli_sname == nullptr) {
                    if (GLOBALS.in_main) {
                        std::cerr << "EXTRA PROF: WARNING unknown function pointer " << fn_ptr << '\n';
                        name_register[ptr - adress_offset] = std::to_string(ptr);
                    } else {
                        name_register[ptr - adress_offset] = "UNKNOWN_INIT_" + std::to_string(ptr);
                    }
                } else {
                    // size_t length = 0;
                    // int status;
                    // char*demangled_name = abi::__cxa_demangle(info.dli_sname, NULL, &length, &status);
                    name_register[ptr - adress_offset] = std::string(info.dli_sname);
                }
            } else {
                if (adress_offset == 0) {
                    adress_offset = base_ptr;
                    GLOBALS.in_main = true;
                } else if (adress_offset != base_ptr) {
                    std::cerr << "EXTRA PROF: WARNING base offset " << base_ptr << " of function pointer " << fn_ptr
                              << " not matching adress offset " << adress_offset << '\n';
                    name_register[ptr - adress_offset] = name_register[ptr - base_ptr];
                }
            }
        } else {
            std::cerr << "EXTRA PROF: WARNING unknown function pointer " << fn_ptr << '\n';
            name_register[ptr - adress_offset] = std::to_string(ptr);
        }
    }
    return push_time(name_register[ptr - adress_offset].c_str(), type);
}

template <>
inline std::tuple<time_point, CallTreeNode *> push_time(char const *name, CallTreeNodeType type) {
    time_point time;
    CUPTI_CALL(cuptiGetTimestamp(&time));
    auto &current_node = GLOBALS.current_node;
    current_node = current_node->findOrAddChild(name, type);
    GLOBALS.timer_stack.push_back(time);
#ifdef EXTRA_PROF_EVENT_TRACE
    auto &ref = cpu_event_stream.emplace_back(time, EventType::START, EventStart{current_node});
    event_stack.push_back(&ref);
#endif
    return {time, current_node};
}

template <typename T>
inline time_point pop_time(T *fn_ptr) {
    auto &name_register = GLOBALS.name_register;
    auto &adress_offset = GLOBALS.adress_offset;
    auto ptr = reinterpret_cast<intptr_t>(fn_ptr);
    if (name_register.find(ptr - adress_offset) == name_register.end()) {
        std::cerr << "EXTRA PROF: WARNING unknown function pointer " << fn_ptr << '\n';
        throw std::runtime_error("EXTRA PROF: ERROR unknown function pointer.");
    }
    return pop_time(name_register[ptr - adress_offset].c_str());
}
template <>
inline time_point pop_time(char const *name) {
    time_point time;
    CUPTI_CALL(cuptiGetTimestamp(&time));
    auto &current_node = GLOBALS.current_node;
    auto duration = time - GLOBALS.timer_stack.back();
    if (current_node->name() == nullptr) {
        std::cerr << "EXTRA PROF: WARNING: accessing calltree root\n";
    }
    if (current_node->name() != name) {
        std::cerr << "EXTRA PROF: ERROR: popping wrong node\n";
    }
    current_node->visits++;
    current_node->duration += duration;
    GLOBALS.timer_stack.pop_back();
    if (current_node->parent() == nullptr) {
        throw std::runtime_error("EXTRA PROF: pop_time: Cannot go to parent of root");
    }
    current_node = current_node->parent();
#ifdef EXTRA_PROF_EVENT_TRACE
    cpu_event_stream.emplace_back(time, EventType::END, EventEnd{event_stack.back()});
    event_stack.pop_back();
#endif
    return time;
}

void show_data_sizes() {
    std::cerr << "EXTRA PROF: Event size: " << sizeof(Event) << '\n';
    std::cerr << "EXTRA PROF: CallTreeNode size: " << sizeof(CallTreeNode) << '\n';
}

} // namespace extra_prof