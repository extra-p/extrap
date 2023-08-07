#include "tests.h"

void test_copies(int scaling) {
    PUSH_RANGE("main->test_copies");
    std::vector<cudaStream_t> stream(5);
    std::vector<std::vector<double, pinned_allocator<double>>> data(stream.size());
    std::vector<double*> gpu_data(stream.size());
    PUSH_RANGE("main->test_copies->init_data");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamCreate(&stream[i]);
        data[i].resize(500000 * scaling, i);
        gpu_data[i] = gpu_new(data[i]);
    }
    POP_RANGE();
    PUSH_RANGE("main->test_copies->phase");
    for (int i = 0; i < stream.size(); ++i) {
        std::string name = "main->test_copies->phase->memcpyHD" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(gpu_data[i], data[i], stream[i], cudaMemcpyKind::cudaMemcpyHostToDevice));
        POP_RANGE();

        name = "main->test_copies->phase->memset" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        cudaMemsetAsync(gpu_data[i], 2 * i, sizeof(double) * data[i].size(), stream[i]);
        POP_RANGE();

        name = "main->test_copies->phase->memcpyDH2-" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        gpuErrchk(cudaMemcpyAsync(data[i], gpu_data[i], stream[i], cudaMemcpyKind::cudaMemcpyDeviceToHost));

        POP_RANGE();
    }
    POP_RANGE();
    PUSH_RANGE("main->test_copies->wait");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamSynchronize(stream[i]);
        std::cout << "Data " << i << ": ";
        for (int e = 0; e < data[i].size(); e += (data[i].size() / 10)) {
            std::cout << data[i][e] << ", ";
        }
        std::cout << "\n" << std::endl;
    }
    POP_RANGE();
    PUSH_RANGE("main->test_copies->clean_up");
    for (int i = 0; i < stream.size(); ++i) {
        cudaStreamDestroy(stream[i]);
        cudaFree(gpu_data[i]);
    }
    POP_RANGE();
    POP_RANGE();
}