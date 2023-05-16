#include "../globals.h"
#include <new>         // placement new
#include <type_traits> // aligned_storage

namespace extra_prof {
const uint64_t lib_enabled_features = EXTRA_PROF_ENABLED_FEATURES;

GlobalState::GlobalState() {
    threads.emplace(pthread_self(), ThreadState{0, &call_tree});
    std::cerr << "EXTRA PROF: Global state initialized" << std::endl;
}

GlobalState GLOBALS;
}
