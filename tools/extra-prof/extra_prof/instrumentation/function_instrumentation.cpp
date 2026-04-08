#include "commons.h"
#include "start_end.h"

namespace extra_prof {
void region_enter(RegionType region_type, RegionID this_fn) {
    using namespace extra_prof;
    if (!extra_prof_globals_initialised) {
        return;
    }
    if (extra_prof_scope_counter > 0) {
        return;
    }
    extra_prof_scope sc;

    // std::cout << "Start: " << this_fn << std::endl;
    if (!GLOBALS.initialised) {
        std::lock_guard<std::mutex> lk(GLOBALS.initialising);

        if (!GLOBALS.initialised) {
            GLOBALS.main_thread = pthread_self();
            initialize();

            GLOBALS.initialised = true;
        }
    }
    // if (std::this_thread::get_id() != GLOBALS.main_thread_id) {
    //     if (!GLOBALS.notMainThreadAlreadyWarned.load(std::memory_order_relaxed)) {
    //         std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
    //         GLOBALS.notMainThreadAlreadyWarned.store(true, std::memory_order_relaxed);
    //     }
    //     return;
    // }
    ThreadState& thread_state = GLOBALS.my_thread_state();
    if (thread_state.depth < GLOBALS.MAX_DEPTH) {
        push_time(region_type, this_fn, CallTreeNodeType::NONE, thread_state);
    }

    thread_state.depth++;
}
void region_exit(RegionType region_type, RegionID this_fn) {
    using namespace extra_prof;
    if (!extra_prof_globals_initialised) {
        return;
    }
    // std::cout << "End: " << this_fn << std::endl;
    if (extra_prof_scope_counter > 0) {
        return;
    }
    extra_prof_scope sc;

    if (GLOBALS.initialised) {
        auto& thread_state = GLOBALS.my_thread_state();
        if (thread_state.depth == 0) {
            throw std::underflow_error("EXTRA PROF: ERROR: Stack depth is already zero.");
        };
        thread_state.depth--;
        if (thread_state.depth < GLOBALS.MAX_DEPTH) {
            pop_time(region_type, this_fn, thread_state);
            if (thread_state.depth == 0) {
                if (GLOBALS.name_register.is_main_function(region_type, this_fn)) {
                    finalize();
                }
            }
        }
        if (GLOBALS.name_register.is_main_function(region_type, this_fn) && thread_state.depth != 0) {
            std::cerr << "EXTRA PROF: WARNING: Found end of main function, but call tree is not at root." << std::endl;
        }
    }
}
} // namespace extra_prof

extern "C" {
EXTRA_PROF_SO_EXPORT void __cyg_profile_func_enter(void* this_fn, void* call_site) {
    extra_prof::region_enter(extra_prof::RegionType::FUNCTION_PTR_REGION, extra_prof::toRegionID(this_fn));
}

EXTRA_PROF_SO_EXPORT void __cyg_profile_func_exit(void* this_fn, void* call_site) {
    extra_prof::region_exit(extra_prof::RegionType::FUNCTION_PTR_REGION, extra_prof::toRegionID(this_fn));
}

typedef struct {
    uint32_t* handle;
    const char* name;
    const char* canonical_name;
    const char* file;
    int begin_lno;
    int end_lno;
    unsigned flags; /* unused */
} __attribute__((aligned(64))) scorep_compiler_region_description;

EXTRA_PROF_SO_EXPORT void scorep_plugin_register_region(const scorep_compiler_region_description* regionDescr) {
    extra_prof::GLOBALS.name_register.add_scorep_region(regionDescr->handle, regionDescr->name,
                                                        regionDescr->canonical_name);
    // std::cerr << "Registered " << *regionDescr->handle << '\n';
}
EXTRA_PROF_SO_EXPORT void scorep_plugin_enter_region(extra_prof::ScorepRegion regionHandle) {
    extra_prof::region_enter(extra_prof::RegionType::SCOREP_REGION, extra_prof::toRegionID(regionHandle));
}
EXTRA_PROF_SO_EXPORT void scorep_plugin_exit_region(extra_prof::ScorepRegion regionHandle) {
    extra_prof::region_exit(extra_prof::RegionType::SCOREP_REGION, extra_prof::toRegionID(regionHandle));
}
}