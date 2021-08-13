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
        m_dimBlock(m_vertexMap.block),
        m_dimGrid(m_vertexMap.grid)
    {
        // allocate map memories
        m_depthMap.allocate();
        m_vertexMap.allocate();
        m_normalMap.allocate();
    }
    Level::~Level()
    {
        // free map memories
        m_depthMap.free();
        m_vertexMap.free();
        m_normalMap.free();
    }
}
