#include "tests.h"

__global__ void test(double *a, size_t size, size_t repeat) {
    size_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    /*/	return;
    } */
    for (size_t r = 0; r < repeat; ++r) {
        for (size_t i = start_idx; i < size; i += blockDim.x * gridDim.x) {
            a[i] += 5;
        }
    }
}

void test_max_parallel(int scaling) {
    PUSH_RANGE("main->test_max_par");
    std::vector<cudaStream_t> stream(5);
    std::vector<std::vector<double, pinned_allocator<double>>> data(stream.size());
    std::vector<double *> gpu_data(stream.size());
    PUSH_RANGE("main->test_max_par->init_data");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamCreate(&stream[i]);
        data[i].resize(50000 * scaling, i);
        gpu_data[i] = gpu_new(data[i]);
    }
    POP_RANGE();
    PUSH_RANGE("main->test_max_par->phase");
    for (int i = 0; i < stream.size(); ++i) {
        std::string name = "main->test_max_par->phase->memcpyHD" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(gpu_data[i], data[i], stream[i], cudaMemcpyKind::cudaMemcpyHostToDevice));
        POP_RANGE();

        name = "main->test_max_par->phase->memset" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        cudaMemsetAsync(gpu_data[i], 2 * i, sizeof(double) * data[i].size(), stream[i]);
        POP_RANGE();
        if (i + 2 < stream.size()) {
            name = "main->test_max_par->phase->memset1+" + std::to_string(i);
            PUSH_RANGE(name.c_str());
            cudaMemsetAsync(gpu_data[i + 1], 2 * i, sizeof(double) * data[i + 1].size(), stream[i + 1]);
            POP_RANGE();
            name = "main->test_max_par->phase->memset2+" + std::to_string(i);
            PUSH_RANGE(name.c_str());
            cudaMemsetAsync(gpu_data[i + 2], 2 * i, sizeof(double) * data[i + 2].size(), stream[i + 2]);
            POP_RANGE();
        }

        name = "main->test_max_par->phase->kernel" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        test<<<2, 1024, 0, stream[i]>>>(gpu_data[i], data[i].size(), 5);
        POP_RANGE();
        name = "main->test_max_par->phase->memcpyDH2-" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(data[i], gpu_data[i], stream[i], cudaMemcpyKind::cudaMemcpyDeviceToHost));

        POP_RANGE();
    }
    POP_RANGE();
    PUSH_RANGE("main->test_max_par->wait");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamSynchronize(stream[i]);
        std::cout << "Data " << i << ": ";
        for (int e = 0; e < data[i].size(); e += (data[i].size() / 10)) {
            std::cout << data[i][e] << ", ";
        }
        std::cout << "\n" << std::endl;
    }
    POP_RANGE();
    PUSH_RANGE("main->test_max_par->clean_up");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamDestroy(stream[i]);
        cudaFree(gpu_data[i]);
    }
    POP_RANGE();
    POP_RANGE();
}