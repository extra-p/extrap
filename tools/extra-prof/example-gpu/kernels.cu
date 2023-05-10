#include "tests.h"

#define DEF_TEST_KERNEL(K)                                                                                             \
    __global__ void test##K(double *a, size_t size, size_t repeat) {                                                   \
        size_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;                                                      \
        /*/	return;                                                                                                    \
        } */                                                                                                           \
        for (size_t r = 0; r < repeat; ++r) {                                                                          \
            for (size_t i = start_idx; i < size; i += blockDim.x * gridDim.x) {                                        \
                a[i] += 5;                                                                                             \
            }                                                                                                          \
            __syncthreads();                                                                                           \
            for (size_t i = start_idx + 1; i < size; i += blockDim.x * gridDim.x) {                                    \
                a[i] += a[i - 1];                                                                                      \
            }                                                                                                          \
            __syncthreads();                                                                                           \
            for (size_t i = start_idx + 2; i < size; i += blockDim.x * gridDim.x) {                                    \
                a[i] *= 0.25;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }

DEF_TEST_KERNEL(0)
DEF_TEST_KERNEL(1)
DEF_TEST_KERNEL(2)
DEF_TEST_KERNEL(3)
DEF_TEST_KERNEL(4)

#ifdef __INTELLISENSE__
#define TEST_KERNEL(K)                                                                                                 \
    case K:                                                                                                            \
        break
#else
#define TEST_KERNEL(K)                                                                                                 \
    case K:                                                                                                            \
        test##K<<<1, 1024, 0, stream>>>(a, size, repeat);                                                              \
        break
#endif // __INTELLISENSE__

void test(size_t i, cudaStream_t stream, double *a, size_t size, size_t repeat) {
    switch (i) {
        TEST_KERNEL(0);
        TEST_KERNEL(1);
        // TEST_KERNEL(2);
        TEST_KERNEL(3);
        TEST_KERNEL(4);
    case 2:
        test2<<<18, 1024, 0, stream>>>(a, size, repeat);
        break;
    }
}

void process(int scaling) {
    PUSH_RANGE("main->process");
    constexpr int big_mem_cpy = 2;
    std::vector<cudaStream_t> stream(5);
    PUSH_RANGE("main->process->stream");
    std::vector<cudaEvent_t> sevent(stream.size());
    std::vector<std::vector<double, pinned_allocator<double>>> data(stream.size());
    std::vector<double *> gpu_data(stream.size());
    PUSH_RANGE("main->process->init_data");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&sevent[i]);
        if (i == big_mem_cpy) {
            data[i].resize(500000 * scaling, 10);
        } else {
            data[i].resize(5000 * scaling, i * 1000);
        }
        gpu_data[i] = gpu_new(data[i]);
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase1");
    for (int i = 0; i < stream.size(); ++i) {
        std::string name = "main->process->phase1->memcpyHD" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(gpu_data[i], data[i], stream[i], cudaMemcpyKind::cudaMemcpyHostToDevice));
        POP_RANGE();
        if (i == big_mem_cpy)
            continue;
        name = "main->process->phase1->kernel" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        test(i, stream[i], gpu_data[i], data[i].size(), 50 * (i + 1));
        POP_RANGE();
        if (i % 2 != 0) {
            std::string name = "main->process->phase1->memcpyDH" + std::to_string(i);
            PUSH_RANGE(name.c_str());
            gpuErrchk(cudaMemcpyAsync(data[i], gpu_data[i], stream[i], cudaMemcpyKind::cudaMemcpyDeviceToHost));
            cudaEventRecord(sevent[i], stream[i]);
            POP_RANGE();
        }
        if (i == 4) {
            PUSH_RANGE("main->process->phase1->memset4");
            cudaMemsetAsync(gpu_data[4], 0, sizeof(double) * data[4].size(), stream[4]);
            POP_RANGE();
        }
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase2");
    for (int i = 0; i < stream.size(); i += 2) {
        std::string name = "main->process->phase2->kernel" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        test(i, stream[i], gpu_data[i], data[i].size(), 50 * (i + 1));
        POP_RANGE();
        name = "main->process->phase2->memcpyDH" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(data[i], gpu_data[i], stream[i], cudaMemcpyKind::cudaMemcpyDeviceToHost));
        POP_RANGE();
        cudaEventRecord(sevent[i], stream[i]);
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase3");
    for (int i = 0; i < stream.size(); ++i) {
        cudaEventSynchronize(sevent[i]);
        std::cout << "Data " << i << ": ";
        for (int e = 0; e < data[i].size(); e += (data[i].size() / 10)) {
            std::cout << data[i][e] << ", ";
        }
        std::cout << "\n" << std::endl;
    }
    POP_RANGE();
    PUSH_RANGE("main->process->clean_up");
    for (int i = 0; i < stream.size(); ++i) {
        cudaEventDestroy(sevent[i]);
        cudaStreamDestroy(stream[i]);
        cudaFree(gpu_data[i]);
    }
    POP_RANGE();
    POP_RANGE();
}