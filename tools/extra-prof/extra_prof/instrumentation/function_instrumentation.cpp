#include "commons.h"
#include "start_end.h"

extern "C" {
EXTRA_PROF_SO_EXPORT void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    using namespace extra_prof;
    if (!extra_prof_globals_initialised) {
        return;
    }
    if (extra_prof_scope_counter > 0) {
        return;
    }
    extra_prof_scope sc;

    // std::cout << "Start: " << this_fn << std::endl;
    if (!GLOBALS.initialised.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(GLOBALS.initialising);

        if (!GLOBALS.initialised.load(std::memory_order_relaxed)) {
            // GLOBALS.main_thread_id = std::this_thread::get_id();
            initialize();

            GLOBALS.initialised.store(true, std::memory_order_release);
        }
    }
    // if (std::this_thread::get_id() != GLOBALS.main_thread_id) {
    //     if (!GLOBALS.notMainThreadAlreadyWarned.load(std::memory_order_relaxed)) {
    //         std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
    //         GLOBALS.notMainThreadAlreadyWarned.store(true, std::memory_order_relaxed);
    //     }
    //     return;
    // }
    if (GLOBALS.my_thread_state().depth < GLOBALS.MAX_DEPTH) {
        push_time(this_fn);
    }
    GLOBALS.my_thread_state().depth++;
}

EXTRA_PROF_SO_EXPORT void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    using namespace extra_prof;
    if (!extra_prof_globals_initialised) {
        return;
    }
    // std::cout << "End: " << this_fn << std::endl;
    if (extra_prof_scope_counter > 0) {
        return;
    }
    extra_prof_scope sc;

    if (GLOBALS.initialised.load(std::memory_order_relaxed)) {
        auto &thread_state = GLOBALS.my_thread_state();
        thread_state.depth--;
        if (thread_state.depth < GLOBALS.MAX_DEPTH) {
            pop_time(this_fn);
            if (thread_state.depth == 0) {
                if (GLOBALS.name_register.is_main_function(this_fn)) {
                    finalize();
                }
            }
        }
    }
}
}