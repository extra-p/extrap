#pragma once
#include "common_types.h"

#include "calltree_node.h"
#include "concurrent_array_list.h"
#include "events.h"
#include <cuda.h>
#include <cupti.h>
#include <deque>
#include <thread>
namespace extra_prof {
namespace cupti {
    struct CorrelationData {
        CallTreeNode* const node = nullptr;
        const pthread_t thread;
        const void* function_ptr = nullptr;

    public:
        CorrelationData(CorrelationData&& other) = default;
        CorrelationData(CallTreeNode* const node_, const pthread_t thread_) : node(node_), thread(thread_) {}
        CorrelationData(CallTreeNode* const node_, const pthread_t thread_, const void* const function_ptr_)
            : node(node_), thread(thread_), function_ptr(function_ptr_) {}
    };
} // namespace cupti
namespace gpu {

    struct State {
        extra_prof::MemoryReusePool<uint8_t, ACTIVITY_RECORD_ALIGNMENT> buffer_pool{1024 * 1024};
        CUpti_SubscriberHandle subscriber;

        ConcurrentArrayList<Event> event_stream;
        int multiProcessorCount;

        ConcurrentMap<uint64_t, cupti::CorrelationData> callpath_correlation;

        const char* MEMSET = "Memset";
        const char* MEMSET_ASYNC = "Memset Async";

        const char* OVERLAP = "OVERLAP";

        std::atomic<void (*)(const CUpti_CallbackData*)> onKernelLaunch = nullptr;

        // HWC State

        // Number of ranges recorded by HWC profiler per session
        uint32_t NUM_HWC_RANGES = 256;

        CUcontext currentContext;

        containers::string chipName;
        std::vector<containers::string> metricNames;
        std::vector<uint8_t> counterDataImagePrefix;
        std::vector<uint8_t> configImage;
        std::vector<uint8_t> counterDataImage;
        std::vector<uint8_t> counterDataScratchBuffer;

        std::mutex kernelLaunchMutex;
        uint32_t totalRangeCounter = 0;
        uint32_t rangeCounter = 0;
        std::vector<uint32_t> rangeToCorrelationId;
    };
} // namespace gpu
} // namespace extra_prof