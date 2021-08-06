#include <iostream>

#define cudaSafeCall(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t result, const char* file, int line, bool abort=true)
{
    if (result != cudaSuccess) 
    {
        std::cerr << "CUDA Assert: " << cudaGetErrorString(result) << "\n"
                  << " *** [" << file << ":" << line << "]" << std::endl;
        if (abort)
            exit(result);
    }
}


__global__ void square(float* data)
{
    int i = threadIdx.x;
    data[i] = data[i] * data[i];
}

int main()
{
    unsigned int number = 100;

    float* h_data;
    float* d_data;

    h_data = (float*)malloc(number * sizeof(float));
    cudaSafeCall(cudaMalloc((void**)&d_data, number * sizeof(float)));
    
    for (unsigned int i = 0; i < number; i++)
        h_data[i] = (float)i;
    
    cudaSafeCall(cudaMemcpy(d_data, h_data, number * sizeof(float), cudaMemcpyHostToDevice));
    square<<<1, number>>>(NULL);
    cudaSafeCall(cudaPeekAtLastError());
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaMemcpy(h_data, d_data, number * sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < number; i++)
        std::cout << h_data[i] << (i % 10 == 9 ? "\n" : "\t");
    std::cout << std::endl;

    free(h_data);
    cudaSafeCall(cudaFree(d_data));

    return 0;
}
