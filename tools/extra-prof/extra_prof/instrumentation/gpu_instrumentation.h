#pragma once

#include "../memory_pool.h"

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

#define GPU_LL_CALL(call)                                                                                              \
    do {                                                                                                               \
        CUresult _status = call;                                                                                       \
        if (_status != CUDA_SUCCESS) {                                                                                 \
            const char *errstr;                                                                                        \
            if (cuGetErrorString(_status, &errstr) != CUDA_SUCCESS) {                                                  \
                fprintf(stderr, "%s:%d: error: while decoding error of function %s.\n", __FILE__, __LINE__, #call);    \
                exit(-1);                                                                                              \
            };                                                                                                         \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);   \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace extra_prof {
namespace gpu::runtime {
    extern void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    extern void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size,
                                            size_t validSize);
};
namespace cupti {

    extern void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                     const void *cbdata);

    void init();

    extern void postprocess_event_stream();
    void finalize();

    extern void write_event_stream(std::filesystem::path filename);
    extern size_t event_stream_size();
    extern size_t cupti_mappings_size();
} // namespace cupti
} // namespace extra_prof