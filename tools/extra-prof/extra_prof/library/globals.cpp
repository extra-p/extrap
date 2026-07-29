#include "../globals.h"

#include <new>         // placement new
#include <type_traits> // aligned_storage

namespace extra_prof {
const uint64_t lib_enabled_features = EXTRA_PROF_ENABLED_FEATURES;
bool extra_prof_globals_initialised = false;

GlobalState::GlobalState() {
    call_tree.flags = CallTreeNodeFlags::ROOT;
    if (!extra_prof_globals_initialised) {
        threads.emplace(pthread_self(), ThreadState{0, &call_tree});
        std::cerr << "EXTRA PROF: Global state initialized" << std::endl;
        extra_prof_globals_initialised = true;
    }
}
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
void validate_instrumentation() {

    for (auto&& [tid, threadState] : GLOBALS.threads) {
        if (threadState.depth != threadState.creation_depth) {
            std::stringstream msg;
            msg << "EXTRA PROF: DEBUG: Thread " << tid << " has not returned to its starting depth. Current position: ";
            for (auto* node = threadState.current_node; node != &GLOBALS.call_tree; node = node->parent()) {
                msg << node->name() << "<-";
            }
            std::cerr << msg.str() << std::endl;
        }
        if (threadState.current_node != threadState.creation_node) {
            std::stringstream msg;
            msg << "EXTRA PROF: DEBUG: Thread " << tid
                << " has not returned to its starting call tree node. Current position: ";
            for (auto* node = threadState.current_node; node != &GLOBALS.call_tree; node = node->parent()) {
                msg << node->name() << "<-";
            }
            std::cerr << msg.str() << std::endl;
        }
    }
}
#endif
GlobalState::~GlobalState() {
#ifdef EXTRA_PROF_DEBUG_INSTRUMENTATION
    validate_instrumentation();
#endif
    extra_prof_globals_initialised = false;
}

thread_local int extra_prof_scope_counter;
GlobalState GLOBALS;

} // namespace extra_prof
