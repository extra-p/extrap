#include "so_specific_functions.h"
#include <cstddef>

extern "C" {
void SO_PRIVATE(calculateMaxActiveBlocksPerMultiprocessor)(int *numBlocks, const void *func, int blockSize,
                                                           size_t dynamicSmemSize);
}

namespace extra_prof::gpu {
void calculateMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
}
