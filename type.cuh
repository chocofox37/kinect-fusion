#pragma once

#include <Eigen/Dense>

#include "map.cuh"

namespace kf
{
    /**
     * @brief Depth data
     * @details Float number (float)
     */
    typedef float Depth;

    /**
     * @brief Vertex position
     * @details 3D vector (Eigen::Vector3f)
     */
    typedef Eigen::Vector3f Vertex;

    /**
     * @brief Normal direction
     * @details 3D vector (Eigen::Vector3f)
     */
    typedef Eigen::Vector3f Normal;

    /**
     * @brief Map of depth data
     * @details 2D array of float number (kf::Map<float>)
     */
    typedef Map<Depth> DepthMap;

    /**
     * @brief Map of vertex position
     * @details 2D array of 3D vector (kf::Map<Eigen::Vector3f>)
     */
    typedef Map<Vertex> VertexMap;

    /**
     * @brief Map of normal direction
     * @details 2D array of 3D vector (kf::Map<Eigen::Vector3f>)
     */
    typedef Map<Normal> NormalMap;
}
