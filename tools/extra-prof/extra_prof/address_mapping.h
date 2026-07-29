#pragma once

#include "common_types.h"
#include "concurrent_map.h"
#include "containers/string.h"
#include "filesystem.h"
#include "memory_pool.h"

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <msgpack.hpp>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace extra_prof {

class NameRegistry {
    std::unordered_map<intptr_t, containers::string> name_register;
    NonReusableBlockPool<containers::string, 1024> names;

    std::array<ConcurrentMap<RegionID, containers::string*>, RegionType::REGIONTYPES_LENGTH> dynamic_name_register;

    std::array<RegionID, RegionType::REGIONTYPES_LENGTH> main_function_ptr;
    ScorepRegion region_counter = 0;
    std::mutex region_mutex;

    containers::string GLOBAL_INIT = "_GLOBAL_INIT";

public:
    containers::string defaultExperimentDirName();
    void create_address_mapping(containers::string output_dir);

    EP_INLINE bool is_main_function(RegionType region_type, RegionID region) {
        return region.comparison_value == main_function_ptr[region_type].comparison_value;
    }

    EP_INLINE bool add_scorep_region(ScorepRegion* handle, const char* name, const char* canonical_name) {
        std::lock_guard lg(region_mutex); // lock is necessary to prevent multiple assignments by different threads
        if (*handle == 0) {
            auto region = ++region_counter;
            auto name_ptr = names.construct(name);
            auto result = dynamic_name_register[SCOREP_REGION].emplace(toRegionID(region), name_ptr);
            if (main_function_ptr[SCOREP_REGION].comparison_value == 0 && strcmp(canonical_name, "main") == 0) {
                main_function_ptr[SCOREP_REGION] = toRegionID(region);
            }
            *handle = region; // needs to be assigned at the end
            return result;
        }
        return false;
    }

    __attribute__((used)) EP_INLINE const char* get_name_ptr(RegionType region_type, RegionID region) {
        if (region_type == FUNCTION_PTR_REGION) {
            auto ptr = region.function_ptr;
            auto iter = name_register.find(reinterpret_cast<intptr_t>(ptr));
            if (iter != name_register.end()) {
                return iter->second.c_str();
            }
            containers::string* name_ptr = nullptr;
            if (dynamic_name_register[FUNCTION_PTR_REGION].visit(region, [&](auto& item) { name_ptr = item.second; })) {
                return name_ptr->c_str();
            }
            Dl_info info;
            int status = dladdr(ptr, &info);
            if (status != 0) {
                intptr_t base_ptr = reinterpret_cast<intptr_t>(info.dli_fbase);
                if (info.dli_sname == nullptr) {
                    name_ptr = &GLOBAL_INIT;
                    dynamic_name_register[FUNCTION_PTR_REGION].emplace(region, &GLOBAL_INIT);
                } else {
                    name_ptr = names.construct(info.dli_sname);
                    dynamic_name_register[FUNCTION_PTR_REGION].emplace(region, name_ptr);
                }

            } else {
                std::cerr << "EXTRA PROF: WARNING unknown function pointer " << region.comparison_value << '\n';
                name_ptr = names.construct(std::to_string(region.comparison_value));
                dynamic_name_register[FUNCTION_PTR_REGION].insert_or_assign(region, name_ptr);
            }
            return name_ptr->c_str();
        } else if (region_type == NAMED_REGION) {
            return region.name;
        } else {
            containers::string* name_ptr = nullptr;
            if (dynamic_name_register[region_type].visit(region, [&](auto& item) { name_ptr = item.second; })) {
                return name_ptr->c_str();
            }
            std::cerr << "EXTRA PROF: WARNING unknown region " << region.comparison_value << " of type " << region_type
                      << '\n';
            name_ptr = names.construct(std::to_string(region.comparison_value));
            dynamic_name_register[region_type].insert_or_assign(region, name_ptr);
            return name_ptr->c_str();
        }
    }

    size_t getByteSize() {
        auto size = name_register.size() * (sizeof(intptr_t) + sizeof(containers::string)) +
                    std::accumulate(name_register.cbegin(), name_register.cend(), 0,
                                    [](size_t size, auto& kv) { return size + kv.second.size(); });
        size += sizeof(dynamic_name_register);
        for (auto& nregister : dynamic_name_register) {
            size += nregister.size() * (sizeof(RegionID) + sizeof(containers::string*)) +
                    std::accumulate(nregister.cbegin(), nregister.cend(), 0,
                                    [](size_t size, auto& kv) { return size + kv.second->size(); });
        }
        size += names.byte_size();
        return size;
    }

    const containers::string& search_for_name(const containers::string& name) {
        for (const auto& [key, value] : name_register)
            if (value == name)
                return value;

        containers::string* name_ptr = nullptr;

        for (auto& dnr : dynamic_name_register) {
            auto continue_search = dnr.visit_while([&](auto& item) {
                if (*(item.second) == name) {
                    name_ptr = item.second;
                    return false;
                }
                return true;
            });
            if (!continue_search) {
                return *name_ptr;
            }
        }

        // for (const auto& [key, value] : dynamic_name_register)
        //     if (value == name)
        //         return value;

        throw std::runtime_error("Name is not registered");
    }
};
} // namespace extra_prof