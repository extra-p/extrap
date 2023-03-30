#include "commons.h"
#include "start_end.h"

extern "C" {
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    using namespace extra_prof;
    // std::cout << "Start: " << this_fn << std::endl;
    if (!GLOBALS.initialised.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(GLOBALS.initialising);

        if (!GLOBALS.initialised.load(std::memory_order_relaxed)) {
            GLOBALS.main_thread_id = std::this_thread::get_id();
            initialize();

            GLOBALS.initialised.store(true, std::memory_order_release);
        }
    }
    if (std::this_thread::get_id() != GLOBALS.main_thread_id) {
        std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
        return;
    }
    if (GLOBALS.depth < GLOBALS.MAX_DEPTH) {
        push_time(this_fn);
    }
    GLOBALS.depth++;
}

void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    using namespace extra_prof;
    // std::cout << "End: " << this_fn << std::endl;

    if (GLOBALS.initialised.load(std::memory_order_relaxed)) {
        if (std::this_thread::get_id() != GLOBALS.main_thread_id) {
            std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
            return;
        }
        GLOBALS.depth--;
        if (GLOBALS.depth < GLOBALS.MAX_DEPTH) {
            pop_time(this_fn);
            if (GLOBALS.depth == 0) {
                if (reinterpret_cast<uintptr_t>(this_fn) - GLOBALS.adress_offset == GLOBALS.main_function_ptr) {
                    finalize();
                }
            }
        }
    }
}
}