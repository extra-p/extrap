#include "commons.h"
#include "start_end.h"

extern "C" {
void __cyg_profile_func_enter(void *this_fn, void *call_site) {
    // std::cout << "Start: " << this_fn << std::endl;
    if (!extra_prof::initialised.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(extra_prof::initialising);

        if (!extra_prof::initialised.load(std::memory_order_relaxed)) {
            extra_prof::main_thread_id = std::this_thread::get_id();
            extra_prof::initialize();

            extra_prof::initialised.store(true, std::memory_order_release);
        }
    }
    if (std::this_thread::get_id() != extra_prof::main_thread_id) {
        std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
        return;
    }
    if (extra_prof::depth < extra_prof::MAX_DEPTH) {
        extra_prof::push_time(this_fn);
    }
    extra_prof::depth++;
}

void __cyg_profile_func_exit(void *this_fn, void *call_site) {
    // std::cout << "End: " << this_fn << std::endl;

    if (extra_prof::initialised.load(std::memory_order_relaxed)) {
        if (std::this_thread::get_id() != extra_prof::main_thread_id) {
            std::cerr << "EXTRA PROF: WARNING: Ignored additional threads.\n";
            return;
        }
        extra_prof::depth--;
        if (extra_prof::depth < extra_prof::MAX_DEPTH) {
            extra_prof::pop_time(this_fn);
            if (extra_prof::depth == 0) {
                if (reinterpret_cast<uintptr_t>(this_fn) == extra_prof::main_function_ptr) {
                    extra_prof::finalize();
                }
            }
        }
    }
}
}