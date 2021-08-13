#include "level.cuh"

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
    const unsigned int cuiN = 10;
    kf::Level level(cuiN, cuiN, {0, 0, 0, 0});

    for (unsigned int uiX = 0; uiX < cuiN; uiX++)
    {
        for (unsigned int uiY = 0; uiY < cuiN; uiY++)
        {
            level.depthMap.at(uiX, uiY) = (float)uiX * uiY;
            std::cout << level.depthMap.at(uiX, uiY) << (uiY + 1 == cuiN ? "\n" : "\t");
        }
    }
    std::cout << std::endl;

    level.depthMap.upload();

    square<<<level.grid, level.block>>>(level.depthMap);
    assertKernel("Failed to compute square");

    level.depthMap.download();

    for (unsigned int uiX = 0; uiX < cuiN; uiX++)
    {
        for (unsigned int uiY = 0; uiY < cuiN; uiY++)
        {
            std::cout << level.depthMap.at(uiX, uiY) << (uiY + 1 == cuiN ? "\n" : "\t");
        }
    }
    std::cout << std::endl;

    return 0;
}
