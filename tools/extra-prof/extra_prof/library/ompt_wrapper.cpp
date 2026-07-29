#include "../globals.h"
#include <dlfcn.h>
#include <execinfo.h>
#include <omp-tools.h>
#include <omp.h>

namespace extra_prof::ompt {

void on_ompt_callback_parallel_begin(ompt_data_t* encountering_task_data, const ompt_frame_t* encountering_task_frame,
                                     ompt_data_t* parallel_data, unsigned int requested_parallelism, int flags,
                                     const void* codeptr_ra) {
    auto& current_thread_state = extra_prof::GLOBALS.my_thread_state();
    parallel_data->ptr = new extra_prof::ThreadState(current_thread_state.depth, current_thread_state.current_node);
}

void on_ompt_callback_parallel_end(ompt_data_t* parallel_data, ompt_data_t* encountering_task_data, int flags,
                                   const void* codeptr_ra) {
    delete reinterpret_cast<extra_prof::ThreadState*>(parallel_data->ptr);
}

void on_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t* parallel_data, ompt_data_t* task_data,
                                    unsigned int team_size, unsigned int thread_num) {
    // uint64_t tid = ompt_get_thread_data()->value;
    if (endpoint != ompt_scope_begin) {
        return;
    }
    if (parallel_data == nullptr) {
        return;
    }
    if (parallel_data->ptr == nullptr) {
        return;
    }
    {
        extra_prof_scope sc;
        auto* parent_thread_state = reinterpret_cast<extra_prof::ThreadState*>(parallel_data->ptr);
        auto& my_thread_state = extra_prof::GLOBALS.my_thread_state();
        my_thread_state.current_node = parent_thread_state->current_node;
        my_thread_state.depth = parent_thread_state->depth;
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
        my_thread_state.creation_depth = my_thread_state.depth;
        my_thread_state.creation_node = my_thread_state.current_node;
#endif
    }
}

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t* tool_data) {

    ompt_set_callback_t ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
    if (ompt_set_callback(ompt_callback_implicit_task, (ompt_callback_t)&on_ompt_callback_implicit_task) ==
        ompt_set_never) {
        std::cerr << "EXTRA PROF: ERROR: Could not register callback for ompt_callback_implicit_task" << std::endl;
        return 0;
    }
    if (ompt_set_callback(ompt_callback_parallel_begin, (ompt_callback_t)&on_ompt_callback_parallel_begin) ==
        ompt_set_never) {
        std::cerr << "EXTRA PROF: ERROR: Could not register callback for ompt_callback_parallel_begin" << std::endl;
        return 0;
    }
    if (ompt_set_callback(ompt_callback_parallel_end, (ompt_callback_t)&on_ompt_callback_parallel_end) ==
        ompt_set_never) {
        std::cerr << "EXTRA PROF: ERROR: Could not register callback for ompt_callback_parallel_end" << std::endl;
        return 0;
    }
    return 1;
}

void ompt_finalize(ompt_data_t* tool_data) {}

ompt_start_tool_result_t start_result{&ompt_initialize, &ompt_finalize, 0};

} // namespace extra_prof::ompt

extern "C" {

EXTRA_PROF_SO_EXPORT ompt_start_tool_result_t* ompt_start_tool(unsigned int omp_version, const char* runtime_version) {

    if (omp_version != _OPENMP) {
        std::cerr << "EXTRA PROF: WARNING: OpenMP runtime version (" << omp_version
                  << ") does not match the compile time version (" << _OPENMP << ") for runtime identifying as "
                  << runtime_version << std::endl;
    }

    return &extra_prof::ompt::start_result;
}
}