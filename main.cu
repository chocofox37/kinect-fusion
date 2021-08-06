#include "type.cuh"

__global__ void square(kf::DepthMap depthMap)
{
    unsigned int uiX = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int uiY = threadIdx.y + blockIdx.y * blockDim.y;

    if (depthMap.inside(uiX, uiY))
    {
        kf::Depth depth = depthMap(uiX, uiY);
        depthMap(uiX, uiY) = depth * depth;
    }
}

int main()
{
    kf::DepthMap depthMap(10, 10);

    depthMap.allocate();

    for (unsigned int uiX = 0; uiX < 10; uiX++)
    {
        for (unsigned int uiY = 0; uiY < 10; uiY++)
        {
            depthMap.at(uiX, uiY) = (float)uiX * uiY;
            std::cout << depthMap.at(uiX, uiY) << (uiY == 9 ? "\n" : "\t");
        }
    }
    std::cout << std::endl;

    depthMap.upload();

    square<<<depthMap.grid, depthMap.block>>>(depthMap);
    assertKernel("Failed to compute square");

    depthMap.download();

    for (unsigned int uiX = 0; uiX < 10; uiX++)
    {
        for (unsigned int uiY = 0; uiY < 10; uiY++)
        {
            std::cout << depthMap.at(uiX, uiY) << (uiY == 9 ? "\n" : "\t");
        }
    }
    std::cout << std::endl;

    depthMap.free();

    return 0;
}
