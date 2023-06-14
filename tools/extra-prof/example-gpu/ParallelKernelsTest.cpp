// ParallelKernelsTest.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "ParallelKernelsTest.h"
#include "library.hpp"
#include "tests.h"

#include <chrono>
#include <thread>

using namespace std;

nvtxDomainHandle_t global::nvtx_domain = {0};
int global::depth = 0;

void thread_func() {
    PUSH_RANGE("thread_func");
    std::cout << "Message from thread" << std::this_thread::get_id() << '\n';
}

void parallel_test() { PUSH_RANGE("parallel_test"); }

int main(int argc, char *argv[]) {
    auto start = std::chrono::steady_clock::now();

    dynamicHello();

    if (argc != 2) {
        printf("Missing arguments.");
        return -1;
    }
#pragma omp parallel for
    for (int i = 0; i < 5; i++) {
        PUSH_RANGE("Testp");
    }
    dynamicHello();
    runDynamicKernel();
    int scaling = std::stoi(argv[1]);
    global::nvtx_domain = nvtxDomainCreateA("de.tu-darmstadt.parallel.extra_prof");
    PUSH_RANGE("main");
    cout << "Starting tests. Scaling: " << scaling << endl;
    for (int i = 0; i < 2; i++) {
        // test_max_parallel(scaling);
        // // std::thread test_thread(thread_func);
        // test_copies(scaling);
        // process(scaling);
        // test_thread.join();
    }

#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        process(scaling);
        PUSH_RANGE("Test parallel GPU");
        parallel_test();
        dynamicHello();
    }
    cout << "Took: "
         << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()
         << endl;
    return 0;
}
