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

#define GPU_CALL(call)                                                                                                 \
    do {                                                                                                               \
        cudaError_t _status = call;                                                                                    \
        if (_status != cudaSuccess) {                                                                                  \
            const char *errstr = cudaGetErrorString(_status);                                                          \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);   \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace extra_prof {
namespace cupti {

    extern void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                     const void *cbdata);

    extern void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    extern void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size,
                                            size_t validSize);

    inline void init() {

        const char *EXTRA_PROF_CUPTI_BUFFER_SIZE = getenv("EXTRA_PROF_CUPTI_BUFFER_SIZE");
        if (EXTRA_PROF_CUPTI_BUFFER_SIZE != nullptr) {
            size_t number = std::stoul(EXTRA_PROF_CUPTI_BUFFER_SIZE);
            if (number < 1024) {

                number = 1024;
            }
            std::cerr << "EXTRA PROF: CUPTI BUFFER SIZE: " << number << '\n';
            GLOBALS.gpu.buffer_pool.initial_buffer_resize(number);
        }

        CUPTI_CALL(cuptiSubscribe(&GLOBALS.gpu.subscriber, (CUpti_CallbackFunc)&on_callback, nullptr));
        CUPTI_CALL(cuptiActivityRegisterCallbacks(&on_buffer_request, &on_buffer_complete));

        cudaDeviceProp prop;
        GPU_CALL(cudaGetDeviceProperties(&prop, 0));
        GLOBALS.gpu.multiProcessorCount = prop.multiProcessorCount;

        CUPTI_CALL(cuptiEnableDomain(1, GLOBALS.gpu.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
    }

    extern void postprocess_event_stream();
    inline void finalize() {
        cudaDeviceSynchronize();
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        cuptiFinalize();
        postprocess_event_stream();
    }

    extern void write_event_stream(std::filesystem::path filename);
    extern size_t event_stream_size();
    extern size_t cupti_mappings_size();
} // namespace cupti
} // namespace extra_prof