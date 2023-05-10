#include "calltree_node.h"
#include "common_types.h"
#include "events.h"
#include <csignal>
#include <vector>
namespace extra_prof {
struct ThreadState {
    uint32_t depth = 0;
    CallTreeNode *current_node;

    std::vector<time_point> timer_stack;
#ifdef EXTRA_PROF_EVENT_TRACE
    std::vector<Event *> event_stack;
#endif

    ThreadState() {
        std::cerr << "EXTRA PROF: Auto creation of threads not allowed\n";
        std::raise(SIGTRAP);
    }
    ThreadState(uint32_t depth_, CallTreeNode *node) : depth(depth_), current_node(node) {}
    ThreadState(const ThreadState &state) = delete;
    ThreadState(ThreadState &&state) = default;

    ThreadState duplicate() const {
        ThreadState state(depth, current_node);
        return state;
    }
};
}