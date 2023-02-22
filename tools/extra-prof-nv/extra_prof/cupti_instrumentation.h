#pragma once

#include "memory_pool.h"
#include <cupti.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>

#define CUPTI_CALL(call)                                                                                               \
    do {                                                                                                               \
        CUptiResult _status = call;                                                                                    \
        if (_status != CUPTI_SUCCESS) {                                                                                \
            const char *errstr;                                                                                        \
            cuptiGetResultString(_status, &errstr);                                                                    \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);   \
            if (_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)                                                  \
                exit(0);                                                                                               \
            else                                                                                                       \
                exit(-1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace extra_prof {
namespace cupti {
    inline extra_prof::MemoryReusePool<uint8_t, ACTIVITY_RECORD_ALIGNMENT> buffer_pool(1024 * 1024);

    inline CUpti_SubscriberHandle subscriber;
    extern void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                     const void *cbdata);

    void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size,
                                     size_t validSize);

    inline uint32_t maxThreadsPerMultiProcessor;
    inline uint32_t maxBlocksPerMultiProcessor;
    inline uint32_t multiProcessorCount;

    inline void init() {

        const char *EXTRA_PROF_CUPTI_BUFFER_SIZE = getenv("EXTRA_PROF_CUPTI_BUFFER_SIZE");
        if (EXTRA_PROF_CUPTI_BUFFER_SIZE != nullptr) {
            size_t number = std::stoul(EXTRA_PROF_CUPTI_BUFFER_SIZE);
            if (number < 1024) {

                number = 1024;
            }
            std::cerr << "EXTRA PROF: CUPTI BUFFER SIZE: " << number << '\n';
            buffer_pool.initial_buffer_resize(number);
        }

        CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)&on_callback, nullptr));
        CUPTI_CALL(cuptiActivityRegisterCallbacks(&on_buffer_request, &on_buffer_complete));

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess)
            printf("%s\n", cudaGetErrorString(err));
        maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        maxBlocksPerMultiProcessor = prop.maxBlocksPerMultiProcessor;
        multiProcessorCount = prop.multiProcessorCount;

        CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
    }

    extern void postprocess_event_stream();
    inline void finalize() {
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        postprocess_event_stream();
    }

    extern void write_event_stream(std::filesystem::path filename);
    extern size_t event_stream_size();
    extern size_t cupti_mappings_size();
} // namespace cupti
} // namespace extra_prof