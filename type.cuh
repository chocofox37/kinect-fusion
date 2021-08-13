#pragma once

#include <Eigen/Dense>

#include "map.cuh"

namespace kf
{
    /**
     * @brief Camera intrinsic parameters (focal length, principal point)
     * @details Struct of 4 float numbers ({float, float, float, float})
     */
    struct Intrinsic
    {
        float fx; /**< Focal length (X direction) */
        float fy; /**< Focal length (Y direction) */
        float cx; /**< Principal point (X coordinate) */
        float cy; /**< Principal point (Y coordinate) */
    };

    /**
     * @brief Depth data
     * @details Float number (float)
     */
    typedef float Depth;

    /**
     * @brief Map of depth data
     * @details 2D array of float number (kf::Map<float>)
     */
    typedef Map<Depth> DepthMap;

    /**
     * @brief Vertex position
     * @details 3D vector (Eigen::Vector3f)
     */
    typedef Eigen::Vector3f Vertex;

    /**
     * @brief Map of vertex position
     * @details 2D array of 3D vector (kf::Map<Eigen::Vector3f>)
     */
    typedef Map<Vertex> VertexMap;

    /**
     * @brief Normal direction
     * @details 3D vector (Eigen::Vector3f)
     */
    typedef Eigen::Vector3f Normal;

    /**
     * @brief Map of normal direction
     * @details 2D array of 3D vector (kf::Map<Eigen::Vector3f>)
     */
    typedef Map<Normal> NormalMap;

    /**
     * @brief Data validity
     * @details Boolean (bool)
     */
    typedef bool Validity;

    /**
     * @brief Map of data validity
     * @details 2D array of boolean (kf::Map<bool>)
     */
    typedef Map<Validity> ValidityMask;
}
