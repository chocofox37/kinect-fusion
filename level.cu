#include "level.cuh"

namespace kf
{
    Level::Level(unsigned int uiWidth, unsigned int uiHeight, Intrinsic intrinsic) :
        m_uiWidth(uiWidth),
        m_uiHeight(uiHeight),
        m_intrinsic(intrinsic),
        m_depthMap(uiWidth, uiHeight),
        m_vertexMap(uiWidth, uiHeight),
        m_normalMap(uiWidth, uiHeight),
        m_validityMask(uiWidth, uiHeight),
        m_dimBlock(m_vertexMap.block),
        m_dimGrid(m_vertexMap.grid)
    {
        // allocate maps
        m_depthMap.allocate();
        m_vertexMap.allocate();
        m_normalMap.allocate();
        m_validityMask.allocate();
    }

    Level::~Level()
    {
        // free maps
        m_depthMap.free();
        m_vertexMap.free();
        m_normalMap.free();
        m_validityMask.free();
    }

    void Level::setDepthMap(const Depth* pDepth)
    {
        // copy data into depth map
        std::copy(pDepth, pDepth + (m_uiWidth * m_uiHeight), m_depthMap.data());

        // upload depth map
        m_depthMap.upload();
    }

    __global__ void __compute_vertex_map(
        Intrinsic intrinsic, DepthMap depthMap, VertexMap vertexMap)
    {
        CUDA_KERNEL_INDEX_2D

        // check out of index
        if (!depthMap.inside(uiX, uiY))
            return;
        
        // get depth and vertex data
        Depth& depth = depthMap(uiX, uiY);
        Vertex& vertex = vertexMap(uiX, uiY);

        // compute vertex position
        vertex(0) = depth * ((float)uiX - intrinsic.cx) / intrinsic.fx;
        vertex(1) = depth * ((float)uiY - intrinsic.cy) / intrinsic.fy;
        vertex(2) = depth;
    }

    void Level::computeVertexMap()
    {
        // compute vertex map by CUDA kernel
        __compute_vertex_map<<<m_vertexMap.grid, m_vertexMap.block>>>(
            m_intrinsic, m_depthMap, m_vertexMap);
        ASSERT_CUDA_KERNEL("Failed to compute vertex map");

#if KF_TEST_DOWNLOAD

        // download vertex map
        m_vertexMap.download();
        
#endif
    }

    __global__ void __compute_normal_map(
        DepthMap depthMap, VertexMap vertexMap, NormalMap normalMap, ValidityMask validityMask)
    {
        CUDA_KERNEL_INDEX_2D

        // check out of index
        if (!depthMap.inside(uiX, uiY, 1))
            return;
        
        // initialize validity
        Validity validity = true;
        
        // check depth data validity
        if (!(depthMap(uiX, uiY) > 0))
            validity = false;
        
        // check depth data validity from neighbors
        if (!(depthMap(uiX + 1, uiY) > 0 && depthMap(uiX - 1, uiY) > 0))
            validity = false;
        if (!(depthMap(uiX, uiY + 1) > 0 && depthMap(uiX, uiY - 1) > 0))
            validity = false;
        
        // check validity
        if (validity)
        {
            // compute normal direction from neighbor vertices
            auto dX = vertexMap(uiX + 1, uiY) - vertexMap(uiX - 1, uiY);
            auto dY = vertexMap(uiX, uiY + 1) - vertexMap(uiX, uiY - 1);
            normalMap(uiX, uiY) = dX.cross(dY).normalized();
        }
        else
        {
            // reset normal direction to zero
            normalMap(uiX, uiY).setZero();
        }

        // set validity
        validityMask(uiX, uiY) = validity;
    }

    void Level::computeNormalMap()
    {
        // compute normal map by CUDA kernel
        __compute_normal_map<<<m_normalMap.grid, m_normalMap.block>>>(
            m_depthMap, m_vertexMap, m_normalMap, m_validityMask);
        ASSERT_CUDA_KERNEL("Failed to compute normal map");

#if KF_TEST_DOWNLOAD

        // download normal map
        m_normalMap.download();

        // download validity mask
        m_validityMask.download();

#endif
    }
}
