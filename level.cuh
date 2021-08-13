#pragma once

#include "type.cuh"

namespace kf
{
    class Level
    {
    private:
        unsigned int m_uiWidth;
        unsigned int m_uiHeight;
        Intrinsic m_intrinsic;
        DepthMap m_depthMap;
        VertexMap m_vertexMap;
        NormalMap m_normalMap;
        dim3 m_dimBlock;
        dim3 m_dimGrid;
    public:
        const unsigned int& width = m_uiWidth;
        const unsigned int& height = m_uiHeight;
        const Intrinsic& intrinsic = m_intrinsic;
        DepthMap& depthMap = m_depthMap;
        VertexMap& vertexMap = m_vertexMap;
        NormalMap& normalMap = m_normalMap;
        dim3& block = m_dimBlock;
        dim3& grid = m_dimGrid;
    private:
        Level() = delete;
    public:
        Level(unsigned int uiWidth, unsigned int uiHeight, Intrinsic intrinsic);
        ~Level();
    };
}
