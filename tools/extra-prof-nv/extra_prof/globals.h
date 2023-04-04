#pragma once
#include "calltree_node.h"
#include "gpu_state.h"
#include "memory_pool.h"
#include <atomic>
#include <deque>
#include <string>
#include <unordered_map>
namespace extra_prof {

#ifndef EXTRA_PROF_MAX_MAX_DEPTH
constexpr uint32_t MAX_MAX_DEPTH = 1000;
#else
constexpr uint32_t MAX_MAX_DEPTH = EXTRA_PROF_MAX_MAX_DEPTH;
#endif

struct GlobalState {
    const uint32_t magic_number = 0x1A2B3C4D;
    std::atomic<bool> initialised = false;
    std::mutex initialising;

    std::filesystem::path output_dir;

    std::unordered_map<intptr_t, std::string> name_register;
    uintptr_t main_function_ptr;

    intptr_t adress_offset = 0;

    std::atomic<bool> in_main = false;
    std::thread::id main_thread_id;
    std::atomic<bool> notMainThreadAlreadyWarned = false;

    uint32_t depth = 0;
    uint32_t MAX_DEPTH = 30;
    std::vector<time_point> timer_stack;

    CallTreeNode call_tree;
    CallTreeNode *current_node = &call_tree;
    NonReusableBlockPool<CallTreeNode, 64> calltree_nodes_allocator;

#ifdef EXTRA_PROF_EVENT_TRACE
    std::deque<Event> cpu_event_stream;
    std::vector<Event *> event_stack;
#endif

    gpu::State gpu;
};
extern GlobalState &GLOBALS;

namespace globals_init {
    static struct GlobalStateInitializer {
        GlobalStateInitializer();
        ~GlobalStateInitializer();
    } globalStateInitializer;
}
}