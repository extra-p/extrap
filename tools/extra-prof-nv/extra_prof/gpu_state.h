#include "calltree_node.h"
#include "events.h"
#include <cupti.h>
#include <deque>
#include <thread>
namespace extra_prof {
namespace cupti {
    struct CorrelationData {
        CallTreeNode *node = nullptr;
        const void *function_ptr = nullptr;

    public:
        CorrelationData() = default;
        CorrelationData(CorrelationData &&other) = default;
        CorrelationData(CallTreeNode *node_) : node(node_) {}
        CorrelationData(CallTreeNode *node_, const void *function_ptr_) : node(node_), function_ptr(function_ptr_) {}
    };
}
namespace gpu {
    struct State {
        extra_prof::MemoryReusePool<uint8_t, ACTIVITY_RECORD_ALIGNMENT> buffer_pool{1024 * 1024};
        CUpti_SubscriberHandle subscriber;

        std::deque<Event> event_stream;
        int multiProcessorCount;

        std::unordered_map<uint64_t, cupti::CorrelationData> callpath_correlation;
        std::atomic<std::thread::id> activity_thread;

        const char *MEMSET = "Memset";
        const char *MEMSET_ASYNC = "Memset Async";

        const char *OVERLAP = "OVERLAP";
    };
}
}