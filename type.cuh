#include <Eigen/Dense>

#include "map.cuh"

namespace kf
{
    typedef float Depth;
    typedef Eigen::Vector3f Vertex;
    typedef Eigen::Vector3f Normal;

    typedef Map<Depth> DepthMap;
    typedef Map<Vertex> VertexMap;
    typedef Map<Normal> NormalMap;
}
