#pragma once

#include <iostream>

#define ASSERT_CXX(result, message) \
__assert_cxx(result, message, __FILE__, __LINE__);

inline void __assert_cxx(bool result, const char* message, const char* file, int line)
{
    if (!result)
    {
        std::cerr << "CXX Assertion: " << message << "\n"
                  << " *** [" << file << ":" << line << "]" << std::endl;
        exit(result);
    }
}

#define ASSERT_CUDA(result, message) \
__assert_cuda(result, message, __FILE__, __LINE__);

inline void __assert_cuda(cudaError_t result, const char* message, const char* file, int line)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Assertion: " << message << " (" << cudaGetErrorString(result) << ")\n"
                  << " *** [" << file << ":" << line << "]" << std::endl;
        exit(result);
    }
}

#define ASSERT_CUDA_KERNEL(message) \
__assert_cuda(cudaGetLastError(), message, __FILE__, __LINE__); \
__assert_cuda(cudaDeviceSynchronize(), message, __FILE__, __LINE__);

#define CUDA_KERNEL_INDEX_2D \
unsigned int uiX = threadIdx.x + blockIdx.x * blockDim.x; \
unsigned int uiY = threadIdx.y + blockIdx.y * blockDim.y;

#define CUDA_KERNEL_INDEX_3D \
unsigned int uiX = threadIdx.x + blockIdx.x * blockDim.x; \
unsigned int uiY = threadIdx.y + blockIdx.y * blockDim.y; \
unsigned int uiZ = threadIdx.z + blockIdx.z * blockDim.z;
