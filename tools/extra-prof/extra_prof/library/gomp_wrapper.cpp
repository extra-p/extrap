#include <omp.h>

struct gomp_team;
struct gomp_taskgroup;

typedef void (*gomp_parallel_fn)(void (*fn)(void*), void* data, unsigned num_threads, unsigned int flags);

namespace extra_prof {
struct WrappedOMPArgs {
    extra_prof::ThreadState thread_state;
    void (*start_routine)(void*);
    void* argument;
    std::atomic<int> references;
};
void wrap_start_omp(void* wrapped_args_ptr) {
    WrappedOMPArgs* wrapped_args = reinterpret_cast<WrappedOMPArgs*>(wrapped_args_ptr);
    pthread_t new_tid = pthread_self();
    {
        extra_prof_scope sc;
        auto& my_thread_state = extra_prof::GLOBALS.my_thread_state();
        auto& caller_thread_state = wrapped_args->thread_state;
        my_thread_state.current_node = caller_thread_state.current_node;
        my_thread_state.depth = caller_thread_state.depth;
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
        my_thread_state.creation_depth = my_thread_state.depth;
        my_thread_state.creation_node = my_thread_state.current_node;
#endif
    }
    auto start_routine = wrapped_args->start_routine;
    auto args = wrapped_args->argument;
    if ((++wrapped_args->references) >= omp_get_num_threads()) {
        delete wrapped_args;
    }

    start_routine(args);
}
} // namespace extra_prof

extern "C" {

EXTRA_PROF_SO_EXPORT void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned int flags) {

    static void* handle = NULL;
    static gomp_parallel_fn old_start = NULL;
    if (!handle) {
        handle = dlopen("libgomp.so.1", RTLD_LAZY);
        old_start = (gomp_parallel_fn)dlsym(handle, "GOMP_parallel");
    }

    extra_prof::WrappedOMPArgs* wrapped_arg =
        new extra_prof::WrappedOMPArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), fn, data, 0};

    old_start(extra_prof::wrap_start_omp, wrapped_arg, num_threads, flags);
}
}