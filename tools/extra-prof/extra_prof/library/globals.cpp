#include "../globals.h"
#include <new>         // placement new
#include <type_traits> // aligned_storage

namespace extra_prof {
const uint64_t lib_enabled_features = EXTRA_PROF_ENABLED_FEATURES;
std::atomic<bool> extra_prof_globals_initialised = false;

GlobalState::GlobalState() {
    threads.emplace(pthread_self(), ThreadState{0, &call_tree});
    std::cerr << "EXTRA PROF: Global state initialized" << std::endl;
    extra_prof_globals_initialised = true;
}

GlobalState::~GlobalState() { extra_prof_globals_initialised = false; }

thread_local int extra_prof_scope_counter;
GlobalState GLOBALS;

}
