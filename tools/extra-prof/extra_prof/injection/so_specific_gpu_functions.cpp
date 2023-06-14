#include "../so_specific_gpu_functions.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace extra_prof {
namespace gpu {
    void calculateMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize,
                                                   size_t dynamicSmemSize) {
        Dl_info info;
        int status = dladdr(func, &info);
        if (status != 0) {
            auto handle = dlopen(info.dli_fname, RTLD_LAZY);
            dlerror();
            void (*private_function)(int *, const void *, int, size_t) =
                (void (*)(int *, const void *, int, size_t))dlsym(
                    handle, SO_PRIVATE_NAME(calculateMaxActiveBlocksPerMultiprocessor));
            if (private_function == nullptr) {
                char *error = dlerror();
                fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, "dlsym",
                        error);
            } else {
                return private_function(numBlocks, func, blockSize, dynamicSmemSize);
            }
        }
        return SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize, dynamicSmemSize);
    }
}
}

extern "C" {
void SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(int *numBlocks, const void *func, int blockSize,
                                                           size_t dynamicSmemSize) {
    cudaError_t _status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSmemSize);
    if (_status != cudaSuccess) {
        const char *errstr = cudaGetErrorString(_status);
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor", errstr);
    }
}
}