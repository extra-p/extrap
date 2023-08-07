#pragma once
#include "address_mapping.h"
#include "calltree_node.h"
#include "common_types.h"
#include "concurrent_map.h"
#ifdef EXTRA_PROF_GPU
#include "globals_gpu.h"
#endif
#include "containers/string.h"
#include "globals_thread.h"
#include "memory_pool.h"
#include <atomic>
#include <deque>
#include <shared_mutex>
#include <string>

namespace ep {
template <typename mutex>
class lock_guard : public std::lock_guard<mutex> {
    const char* m_name;

public:
    lock_guard(mutex& mu) : std::lock_guard<mutex>(mu) {}
    lock_guard(mutex& mu, const char* name) : std::lock_guard<mutex>(mu), m_name(name) {
        std::cout << "LOCK " << m_name << std::endl;
    }

    ~lock_guard() { std::cout << "UNLOCK " << m_name << std::endl; }
};
} // namespace ep
// #define std::lock_guard ep::lockguard

#ifdef EXTRA_PROF_EVENT_TRACE
#define EXTRA_PROF_EVENT_TRACE_ENABLED 1
#else
#define EXTRA_PROF_EVENT_TRACE_ENABLED 0
#endif
#ifdef EXTRA_PROF_GPU
#define EXTRA_PROF_GPU_ENABLED 1
#else
#define EXTRA_PROF_GPU_ENABLED 0
#endif

#define EXTRA_PROF_ENABLED_FEATURES ((EXTRA_PROF_EVENT_TRACE_ENABLED << 0) + (EXTRA_PROF_GPU_ENABLED << 1))

namespace extra_prof {

struct GlobalState {

    const uint32_t magic_number = 0x1A2B3C4D;

    std::atomic<bool> initialised = false;
    std::mutex initialising;

    containers::string output_dir;

    NameRegistry name_register;

    uint32_t MAX_DEPTH = 30;

    CallTreeNode call_tree;

    NonReusableBlockPool<CallTreeNode, 64> calltree_nodes_allocator;

    ConcurrentMap<pthread_t, ThreadState> threads;

    EP_INLINE ThreadState& my_thread_state() {
        static thread_local ThreadState& state = threads[pthread_self()];
        return state;
    }

    GlobalState();
    ~GlobalState();

#ifdef EXTRA_PROF_EVENT_TRACE
    ConcurrentArrayList<Event> cpu_event_stream;
#endif

#ifdef EXTRA_PROF_GPU
    gpu::State gpu;
#endif
};
extern const uint64_t lib_enabled_features;
extern std::atomic<bool> extra_prof_globals_initialised;
extern thread_local int extra_prof_scope_counter;
extern GlobalState GLOBALS;

struct extra_prof_scope {
    extra_prof_scope() { extra_prof_scope_counter++; }
    ~extra_prof_scope() { extra_prof_scope_counter--; }
};
} // namespace extra_prof