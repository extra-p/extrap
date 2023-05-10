#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <vector>

struct global {
    static nvtxDomainHandle_t nvtx_domain;
    static int depth;
};
inline int PUSH_RANGE(const char *message) {
    std::cout << message << std::endl;
    return 0;
}

inline int POP_RANGE() { return 0; }

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T>
T *gpu_new(size_t size = 1) {
    T *gpu_ptr;
    cudaMalloc(&gpu_ptr, sizeof(T) * size);
    return gpu_ptr;
}

template <typename vector_type>
typename vector_type::value_type *gpu_new(const vector_type &reference) {
    typename vector_type::value_type *gpu_ptr;
    cudaMalloc(&gpu_ptr, sizeof(typename vector_type::value_type) * reference.size());
    return gpu_ptr;
}

template <typename T>
cudaError_t cudaMemcpy(T *dst, const std::vector<T> &src, cudaMemcpyKind kind = cudaMemcpyDefault) {
    return cudaMemcpy(dst, src.data(), sizeof(T) * src.size(), kind);
}
template <typename T>
cudaError_t cudaMemcpy(std::vector<T> &dst, const T *src, cudaMemcpyKind kind = cudaMemcpyDefault) {
    return cudaMemcpy(dst.data(), src, sizeof(T) * dst.size(), kind);
}

template <typename vector_type>
cudaError_t cudaMemcpyAsync(typename vector_type::value_type *dst, const vector_type &src, cudaStream_t stream,
                            cudaMemcpyKind kind = cudaMemcpyDefault) {
    return cudaMemcpyAsync(dst, src.data(), sizeof(typename vector_type::value_type) * src.size(), kind, stream);
}
template <typename vector_type>
cudaError_t cudaMemcpyAsync(vector_type &dst, const typename vector_type::value_type *src, cudaStream_t stream,
                            cudaMemcpyKind kind = cudaMemcpyDefault) {
    return cudaMemcpyAsync(dst.data(), src, sizeof(typename vector_type::value_type) * dst.size(), kind, stream);
}

template <typename T>
class pinned_allocator : public std::allocator<T> {
public:
    typedef size_t size_type;
    typedef T *pointer;
    typedef const T *const_pointer;

    template <typename _Tp1>
    struct rebind {
        typedef pinned_allocator<_Tp1> other;
    };

    pointer allocate(size_type n, const void *hint = 0) {
        // fprintf(stderr, "Alloc %d bytes.\n", n * sizeof(T));
        T *ptr;
        gpuErrchk(cudaMallocHost(&ptr, n * sizeof(T)));
        return ptr;
    }

    void deallocate(pointer p, size_type n) {
        // fprintf(stderr, "Dealloc %d bytes (%p).\n", n * sizeof(T), p);
        gpuErrchk(cudaFreeHost(p));
    }

    pinned_allocator() throw() : std::allocator<T>() { fprintf(stderr, "Hello allocator!\n"); }
    pinned_allocator(const pinned_allocator &a) throw() : std::allocator<T>(a) {}
    template <class U>
    pinned_allocator(const pinned_allocator<U> &a) throw() : std::allocator<T>(a) {}
    ~pinned_allocator() throw() {}
};

void process(int scaling);

void test_copies(int scaling);
void test_max_parallel(int scaling);