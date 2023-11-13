#pragma once
#include "common_types.h"
#include "concurrent_map.h"
#include "containers/string.h"
#include "filesystem.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <msgpack.hpp>
#include <mutex>
#include <numeric>
#include <vector>
namespace extra_prof {

class NameRegistry {
    std::unordered_map<intptr_t, containers::string> name_register;
    ConcurrentMap<intptr_t, containers::string> dynamic_name_register;

    static constexpr uintptr_t adress_offset = 0;

    uintptr_t main_function_ptr = 0;

public:
    containers::string defaultExperimentDirName();
    void create_address_mapping(containers::string output_dir);

    EP_INLINE containers::string* check_ptr(const void* fn_ptr) {
        auto ptr = reinterpret_cast<intptr_t>(fn_ptr);
        auto iter = name_register.find(ptr - adress_offset);
        if (iter != name_register.end()) {
            return &iter->second;
        }
        auto name_ptr = dynamic_name_register.try_get(ptr - adress_offset);
        if (name_ptr == nullptr) {
            std::cerr << "EXTRA PROF: WARNING: unknown function pointer " << fn_ptr << '\n';
            // throw std::runtime_error("EXTRA PROF: ERROR: unknown function pointer.");
        }
        return name_ptr;
    }

    EP_INLINE bool is_main_function(void* this_fn) {
        return reinterpret_cast<uintptr_t>(this_fn) - adress_offset == main_function_ptr;
    }

    EP_INLINE containers::string& get_name_ptr(const void* fn_ptr) {
        auto ptr = reinterpret_cast<intptr_t>(fn_ptr);
        auto iter = name_register.find(ptr - adress_offset);
        if (iter != name_register.end()) {
            return iter->second;
        }
        auto dynamic_name = dynamic_name_register.try_get(ptr - adress_offset);
        if (dynamic_name != nullptr) {
            return *dynamic_name;
        }
        Dl_info info;
        int status = dladdr(fn_ptr, &info);
        if (status != 0) {
            intptr_t base_ptr = reinterpret_cast<intptr_t>(info.dli_fbase);
            if (info.dli_sname == nullptr) {
                dynamic_name_register.try_emplace(ptr - adress_offset, "_GLOBAL_INIT");
            } else {
                dynamic_name_register.try_emplace(ptr - adress_offset, info.dli_sname);
            }

        } else {
            std::cerr << "EXTRA PROF: WARNING unknown function pointer " << fn_ptr << '\n';
            dynamic_name_register[ptr - adress_offset] = std::to_string(ptr);
        }
        return dynamic_name_register[ptr - adress_offset];
    }

    size_t getByteSize() {
        return name_register.size() * (sizeof(intptr_t) + sizeof(containers::string)) +
               std::accumulate(name_register.cbegin(), name_register.cend(), 0,
                               [](size_t size, auto& kv) { return size + kv.second.size(); }) +
               dynamic_name_register.size() * (sizeof(intptr_t) + sizeof(containers::string)) +
               std::accumulate(dynamic_name_register.cbegin(), dynamic_name_register.cend(), 0,
                               [](size_t size, auto& kv) { return size + kv.second.size(); });
    }
};
} // namespace extra_prof