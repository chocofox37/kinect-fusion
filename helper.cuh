#pragma once

#include <iostream>

#define assertCxx(result, message) \
{ \
    _assertCxx((result), (message), __FILE__, __LINE__); \
}

inline void _assertCxx(bool result, const char *message,
                       const char* file, int line, bool abort = true)
{
    if (!result)
    {
        std::cerr << "CXX Assertion: " << message << "\n"
                  << " *** [" << file << ":" << line << "]" << std::endl;
        if (abort) exit(result);
    }
}

#define assertCuda(result, message) \
{ \
    _assertCuda((result), (message), __FILE__, __LINE__); \
}

inline void _assertCuda(cudaError_t result, const char* message,
                        const char* file, int line, bool abort = true)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Assertion: " << message << " (" << cudaGetErrorString(result) << ")\n"
                  << " *** [" << file << ":" << line << "]" << std::endl;
        if (abort) exit(result);
    }
}

#define assertKernel(message) \
{ \
    _assertCuda(cudaGetLastError(), (message), __FILE__, __LINE__); \
    _assertCuda(cudaDeviceSynchronize(), (message), __FILE__, __LINE__); \
}
