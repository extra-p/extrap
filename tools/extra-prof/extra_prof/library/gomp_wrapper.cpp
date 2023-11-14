#include "../globals.h"
#include <atomic>
#include <dlfcn.h>
#include <execinfo.h>
#include <omp.h>

struct gomp_team;
struct gomp_taskgroup;

typedef void (*gomp_parallel_fn)(void (*fn)(void*), void* data, unsigned num_threads, unsigned int flags);
typedef void (*gomp_parallel_start_fn)(void (*fn)(void*), void* data, unsigned num_threads);
typedef void (*gomp_parallel_loop_fn)(void (*fn)(void*), void* data, unsigned num_threads, long start, long end,
                                      long incr, long chunk_size, unsigned flags);
typedef void (*gomp_parallel_loop_start_fn)(void (*fn)(void*), void* data, unsigned num_threads, long start, long end,
                                            long incr, long chunk_size);

namespace extra_prof::wrappers {
struct WrappedGOMPArgs {
    extra_prof::ThreadState thread_state;
    void (*start_routine)(void*);
    void* argument;
    std::atomic<int> references;
};
void wrap_start_gomp(void* wrapped_args_ptr) {
    WrappedGOMPArgs* wrapped_args = reinterpret_cast<WrappedGOMPArgs*>(wrapped_args_ptr);
    pthread_t new_tid = pthread_self();
    if (new_tid != GLOBALS.main_thread) {
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
    if ((wrapped_args->references.fetch_add(1, std::memory_order_acq_rel) + 1) >= omp_get_num_threads()) {
        delete wrapped_args;
    }

    start_routine(args);
}
} // namespace extra_prof::wrappers

#define EXTRA_PROF_GOMP_PARALLEL_LOOP(type)                                                                            \
    void GOMP_parallel_loop_##type(void (*fn)(void*), void* data, unsigned num_threads, long start, long end,          \
                                   long incr, long chunk_size, unsigned flags) {                                       \
        static void* handle = NULL;                                                                                    \
        static gomp_parallel_loop_fn old_start = NULL;                                                                 \
        if (!handle) {                                                                                                 \
            handle = dlopen("libgomp.so.1", RTLD_LAZY);                                                                \
            old_start = (gomp_parallel_loop_fn)dlsym(handle, "GOMP_parallel_loop" #type);                              \
        }                                                                                                              \
        extra_prof::wrappers::WrappedGOMPArgs* wrapped_arg =                                                           \
            new extra_prof::wrappers::WrappedGOMPArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), fn, data, 0}; \
        old_start(extra_prof::wrappers::wrap_start_gomp, wrapped_arg, num_threads, start, end, incr, chunk_size,       \
                  flags);                                                                                              \
    }

#define EXTRA_PROF_GOMP_PARALLEL_LOOP_START(type)                                                                      \
    void GOMP_parallel_loop_##type##_start(void (*fn)(void*), void* data, unsigned num_threads, long start, long end,  \
                                           long incr, long chunk_size) {                                               \
        static void* handle = NULL;                                                                                    \
        static gomp_parallel_loop_start_fn old_start = NULL;                                                           \
        if (!handle) {                                                                                                 \
            handle = dlopen("libgomp.so.1", RTLD_LAZY);                                                                \
            old_start = (gomp_parallel_loop_start_fn)dlsym(handle, "GOMP_parallel_loop" #type "_start");               \
        }                                                                                                              \
        extra_prof::wrappers::WrappedGOMPArgs* wrapped_arg =                                                           \
            new extra_prof::wrappers::WrappedGOMPArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), fn, data, 0}; \
        old_start(extra_prof::wrappers::wrap_start_gomp, wrapped_arg, num_threads, start, end, incr, chunk_size);      \
    }

extern "C" {

EXTRA_PROF_SO_EXPORT void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned int flags) {

    static void* handle = NULL;
    static gomp_parallel_fn old_start = NULL;
    if (!handle) {
        handle = dlopen("libgomp.so.1", RTLD_LAZY);
        old_start = (gomp_parallel_fn)dlsym(handle, "GOMP_parallel");
    }

    auto duplicate = extra_prof::GLOBALS.my_thread_state().duplicate();

    extra_prof::wrappers::WrappedGOMPArgs* wrapped_arg =
        new extra_prof::wrappers::WrappedGOMPArgs{std::move(duplicate), fn, data, 0};

    old_start(extra_prof::wrappers::wrap_start_gomp, wrapped_arg, num_threads, flags);
}

EXTRA_PROF_SO_EXPORT void GOMP_parallel_start(void (*fn)(void*), void* data, unsigned num_threads) {

    static void* handle = NULL;
    static gomp_parallel_start_fn old_start = NULL;
    if (!handle) {
        handle = dlopen("libgomp.so.1", RTLD_LAZY);
        old_start = (gomp_parallel_start_fn)dlsym(handle, "GOMP_parallel_start");
    }

    extra_prof::wrappers::WrappedGOMPArgs* wrapped_arg =
        new extra_prof::wrappers::WrappedGOMPArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), fn, data, 0};

    old_start(extra_prof::wrappers::wrap_start_gomp, wrapped_arg, num_threads);
}

EXTRA_PROF_SO_EXPORT void GOMP_parallel_reductions(void (*fn)(void*), void* data, unsigned num_threads,
                                                   unsigned int flags) {

    static void* handle = NULL;
    static gomp_parallel_fn old_start = NULL;
    if (!handle) {
        handle = dlopen("libgomp.so.1", RTLD_LAZY);
        old_start = (gomp_parallel_fn)dlsym(handle, "GOMP_parallel_reductions");
    }

    extra_prof::wrappers::WrappedGOMPArgs* wrapped_arg =
        new extra_prof::wrappers::WrappedGOMPArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), fn, data, 0};

    old_start(extra_prof::wrappers::wrap_start_gomp, wrapped_arg, num_threads, flags);
}

EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(static);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(dynamic);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(guided);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(runtime);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(nonmonotonic_dynamic);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(nonmonotonic_guided);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(nonmonotonic_runtime);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP(maybe_nonmonotonic_runtime);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP_START(static);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP_START(dynamic);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP_START(guided);
EXTRA_PROF_SO_EXPORT EXTRA_PROF_GOMP_PARALLEL_LOOP_START(runtime);
}