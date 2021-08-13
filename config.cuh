#pragma once

/**
 * @brief Download device(GPU) data to test something
 * @details Cause delays to download all data from Map and Volume class
 */
#define KF_TEST_DOWNLOAD        true

/**
 * @brief Size of CUDA block in X direction of Map class
 * @details Depending on the max number of threads in a block of GPU (512, 1024, or 2048)
 */
#define KF_MAP_BLOCK_SIZE_X     32

/**
 * @brief Size of CUDA block in Y direction of Map class
 * @details Depending on the max number of threads in a block of GPU (512, 1024, or 2048)
 */
#define KF_MAP_BLOCK_SIZE_Y     32
