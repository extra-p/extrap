#include <cuda_runtime.h>
#include <iostream>

constexpr size_t max_offset = 8000; // Not higher than 8190 for unrolling to work

template <typename T>
__global__ void store_kernel(T *data, T *data2, T *data3, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        auto k = 0.2387693847593857;
#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            data[i + offset] = k;
            data2[i + offset] = k;
            data3[i + offset] = k;
        }
    }
}

template <typename T>
__global__ void load_and_sum_kernel(T *data, T *data2, T *data3, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {

#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += data[i + offset] + data2[i + offset] + data3[i + offset];
        }
        data[i] = k;
    }
}
template <typename T>
__global__ void load_and_sum_kernel_iter(T *data, T *data2, T *data3, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {

#pragma unroll 1
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += data[i + offset] + data2[i + offset] + data3[i + offset];
        }
        data[i] = k;
    }
}
template <typename T>
__global__ void load_and_multiply_kernel(T *data, T *data2, T *data3, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {

#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k *= data[i + offset] * data2[i + offset] * data3[i + offset];
        }
        data[i] = k;
    }
}

template <typename T>
__global__ void iterate_kernel(T *data, size_t size, bool perform = false)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
#pragma unroll 1
        for (size_t r = 0; r < 10; r++)
        {
#pragma unroll 1
            for (size_t offset = 0; offset < max_offset; offset++)
            {
                if (data[i + offset] > 0.5)
                {
                    k += data[i + offset];
                }
            }
        }

        data[i] = k;
    }
}
template <typename T>
__global__ void multiply_kernel(T *data, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        T t = data[i + 1];
        T r = data[i + 2];
#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k *= t * ((T)offset) * ((T)2.3) * r * ((T)1.3);
        }
        data[i] = k;
    }
}

template <typename T>
__global__ void multiply_add_kernel(T *data, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        T t = data[i + 1];
        T r = data[i + 2];
#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k *= t * ((T)offset) + ((T)2.3) + r * ((T)1.3);
        }
        data[i] = k;
    }
}

template <typename T>
__global__ void add_kernel(T *data, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        T t = data[i + 1];
        T r = data[i + 2];
#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += t + ((T)offset) + ((T)2.3) + r + ((T)1.3);
        }
        data[i] = k;
    }
}

template <typename T>
void load_and_divide_kernel(T *data, size_t size)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {

#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k /= data[i + offset];
        }
        data[i] = k;
    }
}
template <typename DATA_TYPE>
__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i > 1) && (i < (n - 1)))
    {
        B[i] = 0.33333f * (A[i - 1] + A[i] + A[i + 1]);
    }
}

template <typename DATA_TYPE>
__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j > 1) && (j < (n - 1)))
    {
        A[j] = B[j];
    }
}
template <typename T>
__global__ void multiply_add_kernel2(T *data, size_t size)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    double k = 0;

    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        float t = data[(i + 1)];
        double r = data[i + 2];
#pragma unroll
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += t * ((float)offset) + 2.3f + r * 1.3f;
        }
        data[i] = k;
    }
}
template <typename T>
__global__ void multiply_add_kernel_iter(T *data, size_t size)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    double k = 0;

    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        float t = data[(i + 1)];
        double r = data[i + 2];
#pragma unroll 1
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += t * ((float)offset) + 2.3f + r * 1.3f;
        }
        data[i] = k;
    }
}
template <typename T>
__global__ void multiply_add_kernel_default(T *data, size_t size)
{
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    double k = 0;

    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {
        k = data[i];
        float t = data[(i + 1)];
        double r = data[i + 2];
        for (size_t offset = 0; offset < max_offset; offset++)
        {
            k += t * ((float)offset) + 2.3f + r * 1.3f;
        }
        data[i] = k;
    }
}

template <typename T>
__global__ void test_kernel(T *data, T *data2, T *data3, size_t size)
{

    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    T k = 0;
    for (size_t i = idx; i < size - max_offset; i += blockDim.x * gridDim.x)
    {

        for (size_t offset = 0; offset < max_offset / 3; offset++)
        {
            k += data[i + offset] * data2[i + offset * 2] + data3[i + offset * 3];
            if (k > 4.2)
            {
                k -= 20;
            }
            else
            {
                k *= 2;
                k -= data2[i + offset * 2];
            }
            data[i + offset] = k;
        }
        data[i] = k;
    }
}

__device__ float euclideanLen(float a, float b, float d)
{
    float mod = (b - a) * (b - a);

    return __expf(-mod / (2.f * d * d));
}
template <typename T>
__global__ void d_bilateral_filter(T *od, int w, int h, float e_d, int r)
{
    double cGaussian[64] = {
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.01,
        0.0075,
        0.005,
        0.0025,
        0.005,
        0.01,
        0.015,
        0.02,
        0.02,
        0.015,
        0.01,
        0.005,
        0.0075,
        0.015,
        0.0225,
        0.03,
        0.03,
        0.0225,
        0.015,
        0.0075,
        0.01,
        0.02,
        0.03,
        0.04,
        0.04,
        0.03,
        0.02,
        0.01,
        0.01,
        0.02,
        0.03,
        0.04,
        0.04,
        0.03,
        0.02,
        0.01,
        0.0075,
        0.015,
        0.0225,
        0.03,
        0.03,
        0.0225,
        0.015,
        0.0075,
        0.005,
        0.01,
        0.015,
        0.02,
        0.02,
        0.015,
        0.01,
        0.005,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.01,
        0.0075,
        0.005,
        0.0025,
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x / 32;
    int y = threadIdx.x % 32;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = 0.f;
    float center = od[y * w + x];

    for (int i = 0; i <= r; i++)
    {
        for (int j = 0; j <= r; j++)
        {
            float curPix = od[(y + i) * w + x + j];
            factor = cGaussian[i + r] * cGaussian[j + r] * // domain factor
                     euclideanLen(curPix, center, e_d);    // range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = t / sum;
}

void checkError(cudaError_t cerror = cudaGetLastError())
{
    if (cerror != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cerror) << std::endl;
    }
}

template <typename T>
void runKernels(size_t size,
                int tbSize, int gSize)
{
    T *gpuData;

    checkError(cudaMalloc(&gpuData, size * sizeof(T)));

    int minGridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, store_kernel<T>);
    store_kernel<T><<<gSize, blockSize>>>(gpuData, gpuData, gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, load_and_sum_kernel<T>);
    load_and_sum_kernel<T><<<gSize, blockSize>>>(gpuData, gpuData, gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, load_and_sum_kernel_iter<T>);
    load_and_sum_kernel_iter<T><<<gSize, blockSize>>>(gpuData, gpuData, gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, load_and_multiply_kernel<T>);
    load_and_multiply_kernel<T><<<gSize, blockSize>>>(gpuData, gpuData, gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiply_kernel<T>);
    multiply_kernel<T><<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add_kernel<T>);
    add_kernel<T><<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiply_add_kernel<T>);
    multiply_add_kernel<T><<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, runJacobiCUDA_kernel1<T>);
    runJacobiCUDA_kernel1<T><<<gSize, blockSize>>>(size, gpuData, gpuData);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, runJacobiCUDA_kernel2<T>);
    runJacobiCUDA_kernel2<T><<<gSize, blockSize>>>(size, gpuData, gpuData);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiply_add_kernel2<T>);
    multiply_add_kernel2<<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, iterate_kernel<T>);
    iterate_kernel<T><<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiply_add_kernel_iter<T>);
    multiply_add_kernel_iter<<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiply_add_kernel_default<T>);
    multiply_add_kernel_default<<<gSize, blockSize>>>(gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, test_kernel<T>);
    test_kernel<T><<<gSize, blockSize>>>(gpuData, gpuData, gpuData, size);
    checkError();
    checkError(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, d_bilateral_filter<T>);
    d_bilateral_filter<T><<<gSize, blockSize>>>(gpuData, size / 42 - 10, 32, 15, 10);
    checkError();
    checkError(cudaDeviceSynchronize());

    // load_and_divide_kernel<T><<<gSize, tbSize>>>(gpuData, size);
    // checkError(cudaDeviceSynchronize());

    checkError(cudaFree(gpuData));
}

int main(int argc, char const *argv[])
{
    if (argc <= 1)
    {
        int mpCount;
        checkError(cudaDeviceGetAttribute(&mpCount, cudaDevAttrMultiProcessorCount, 0));
        std::cout << mpCount << " MultiProcessorCount\n";
        return 0;
    }

    int blockSize = 0;
    int gridSize = std::stoi(argv[1]);
    int size_factor = 1024;

    if (argc >= 3)
    {
        size_factor = std::max(size_factor, std::stoi(argv[2]));
    }
    size_t size = size_factor * gridSize + max_offset;
    std::cout << "Size: " << size << " Grid size: " << gridSize << "\n";

    runKernels<float>(size, blockSize, gridSize);
    runKernels<double>(size, blockSize, gridSize);
    runKernels<int>(size, blockSize, gridSize);
    runKernels<int64_t>(size, blockSize, gridSize);
    return 0;
}
