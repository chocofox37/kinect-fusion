#pragma once

/**
 * @brief Size of CUDA block in X direction of Map class
 * @details Depending on the max number of threads in a block of GPU (512, 1024, or 2048)
 */
#define MAP_BLOCK_SIZE_X     32

/**
 * @brief Size of CUDA block in Y direction of Map class
 * @details Depending on the max number of threads in a block of GPU (512, 1024, or 2048)
 */
#define MAP_BLOCK_SIZE_Y     32
