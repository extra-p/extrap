#pragma once

#include <iostream>

#include <cstring>
#include <string>
#include <vector>

struct global {
    static int depth;
};
inline int PUSH_RANGE(const char *message) {
    // std::cout << message << std::endl;
    return 0;
}

inline int POP_RANGE() { return 0; }

void test(size_t i, int stream, std::vector<double> &a, size_t size, size_t repeat) {
    for (size_t r = 0; r < repeat; ++r) {
        for (size_t i = 0; i < size; i++) {
            a[i] += 5;
        }

        for (size_t i = 0 + 1; i < size; i++) {
            a[i] += a[i - 1];
        }

        for (size_t i = 0 + 2; i < size; i++) {
            a[i] *= 0.25;
        }
    }
}

void process(int scaling) {
    PUSH_RANGE("main->process");
    constexpr int big_mem_cpy = 2;
    std::vector<int> stream(5);
    PUSH_RANGE("main->process->stream");
    std::vector<std::vector<double>> data(stream.size());
    PUSH_RANGE("main->process->init_data");
    for (int i = 0; i < stream.size(); ++i) {
        if (i == big_mem_cpy) {
            data[i].resize(5000 * scaling, 10);
        } else {
            data[i].resize(500 * scaling, i * 1000);
        }
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase1");
    for (int i = 0; i < stream.size(); ++i) {
        std::string name = "main->process->phase1->memcpyHD" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        POP_RANGE();
        if (i == big_mem_cpy)
            continue;
        name = "main->process->phase1->kernel" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        test(i, stream[i], data[i], data[i].size(), 5 * (i + 1));
        POP_RANGE();
        if (i % 2 != 0) {
            std::string name = "main->process->phase1->memcpyDH" + std::to_string(i);
            PUSH_RANGE(name.c_str());
            POP_RANGE();
        }
        if (i == 4) {
            PUSH_RANGE("main->process->phase1->memset4");
            memset(data[4].data(), 0, sizeof(double) * data[4].size());
            POP_RANGE();
        }
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase2");
    for (int i = 0; i < stream.size(); i += 2) {
        std::string name = "main->process->phase2->kernel" + std::to_string(i);
        PUSH_RANGE(name.c_str());
        test(i, stream[i], data[i], data[i].size(), 5 * (i + 1));
        POP_RANGE();
    }
    POP_RANGE();
    PUSH_RANGE("main->process->phase3");
    for (int i = 0; i < stream.size(); ++i) {
        std::cout << "Data " << i << ": ";
        for (int e = 0; e < data[i].size(); e += (data[i].size() / 10)) {
            std::cout << data[i][e] << ", ";
        }
        std::cout << "\n" << std::endl;
    }
    POP_RANGE();
    POP_RANGE();
}

void test_copies(int scaling);
void test_max_parallel(int scaling);