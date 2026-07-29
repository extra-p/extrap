#include "../so_specific_gpu_functions.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace extra_prof {
namespace gpu {
    void calculateMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize,
                                                   size_t dynamicSmemSize) {
        /**
         * Calls cudaOccupancyMaxActiveBlocksPerMultiprocessor indirectly, because the command has to be called in the
         * same lib/executable that also contains the kernel, otherwise it does not work.
         */

        Dl_info info;
        int status = dladdr(func, &info);
        if (status != 0) {

            // if func is in the same lib/executable as this function call
            // SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor) directly
            Dl_info this_info;
            int this_status = dladdr((void*)calculateMaxActiveBlocksPerMultiprocessor, &this_info);
            if (this_status != 0) {
                if (info.dli_fbase == this_info.dli_fbase) {
                    return SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize,
                                                                                 dynamicSmemSize);
                }
            }

            // else call it via dynamic function resolution
            auto handle = dlopen(info.dli_fname, RTLD_LAZY);
            dlerror();
            void (*private_function)(int*, const void*, int, size_t) = (void (*)(int*, const void*, int, size_t))dlsym(
                handle, SO_PRIVATE_NAME(calculateMaxActiveBlocksPerMultiprocessor));
            if (private_function == nullptr) {
                char* error = dlerror();
                fprintf(stderr, "%s:%d: error: function %s failed with error %s when loading: %s.\n", __FILE__,
                        __LINE__, "dlsym", error, info.dli_fname);
            } else {
                return private_function(numBlocks, func, blockSize, dynamicSmemSize);
            }
        }
        return SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize, dynamicSmemSize);
    }
} // namespace gpu
} // namespace extra_prof

extern "C" {
void SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(int* numBlocks, const void* func, int blockSize,
                                                           size_t dynamicSmemSize) {
    cudaError_t _status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSmemSize);
    if (_status != cudaSuccess) {
        const char* errstr = cudaGetErrorString(_status);
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor", errstr);
    }
}
}