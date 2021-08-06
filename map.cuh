#pragma once

#include "setting.cuh"
#include "helper.cuh"

namespace kf
{
    /**
     * @brief 2D array class to save data between host(CPU) & device(GPU)
     * 
     * @tparam T Type of data to store each map pixel
     */
    template<typename T>
    class Map
    {
    private:
        unsigned int m_uiWidth;     /**< Width (Number of pixels in X direction) */
        unsigned int m_uiHeight;    /**< Height (Number of pixels in Y direction) */
        dim3 m_dimBlock;    /**< Dimension of block to run CUDA kernel function */
        dim3 m_dimGrid;     /**< Dimension of grid to run CUDA kernel function */
        T* h_pData;     /**< Pointer of data from host(CPU) */
        T* d_pData;     /**< Pointer of data from device(GPU) */

    public:

        /**
         * @brief Width
         * @details Number of pixels in X direction
         */
        const unsigned int& width = m_uiWidth;

        /**
         * @brief Height
         * @details Number of pixels in Y direction
         */
        const unsigned int& height = m_uiHeight;

        /**
         * @brief Dimension of block
         * @details Need to run CUDA kernel function
         */
        const dim3& block = m_dimBlock;

        /**
         * @brief Dimension of grid
         * @details Need to run CUDA kernel function
         */
        const dim3& grid = m_dimGrid;

    private:

        /**
         * @brief Construct a new Map object
         * @details Disable default constructor
         */
        Map() = delete;

    public:
        /**
         * @brief Construct a new Map object
         * 
         * @param uiWidth Width (Number of pixels in X direction)
         * @param uiHeight Height (Number of pixels in Y direction)
         */
        Map(unsigned int uiWidth, unsigned int uiHeight) :
            m_uiWidth(uiWidth),
            m_uiHeight(uiHeight),
            m_dimBlock(MAP_BLOCK_DIM_X, MAP_BLOCK_DIM_Y),
            m_dimGrid((uiWidth + MAP_BLOCK_DIM_X - 1) / MAP_BLOCK_DIM_X,
                      (uiHeight + MAP_BLOCK_DIM_Y - 1) / MAP_BLOCK_DIM_Y),
            h_pData(nullptr),
            d_pData(nullptr)
        {
            // do nothing
        }

        /**
         * @brief Allocate host(CPU) & device(GPU) memory
         * @details Must call once before using this class (Memory allocation)
         */
        void allocate()
        {
            // check memory allocation state
            assertCxx(!h_pData && !d_pData, "Failed to allocate memory (already allocated)")

            // allocate host and device memory
            assertCxx(h_pData = new T[m_uiWidth * m_uiHeight],
                      "Failed to allocate host memory");
            assertCuda(cudaMalloc((void**)&d_pData, m_uiWidth * m_uiHeight * sizeof(T)),
                       "Failed to allocate device memory");
        }

        /**
         * @brief Free host(CPU) & device(GPU) memory
         * @details Must call once after using this class (Memory leak)
         */
        void free()
        {
            // check memory allocation state
            assertCxx(h_pData && d_pData, "Failed to free memory (never allocated)")

            // free host and device memory
            delete[] h_pData;
            assertCuda(cudaFree(d_pData), "Failed to free device memory");
        }

        /**
         * @brief Copy data from host(CPU) to device(GPU)
         * 
         * @param uiSize Size of data to copy (0: Copy all)
         */
        void upload(unsigned int uiSize = 0)
        {
            // set size to copy all data
            if (uiSize == 0) uiSize = m_uiWidth * m_uiHeight * sizeof(T);

            // copy data from host to device
            assertCuda(cudaMemcpy(d_pData, h_pData, uiSize, cudaMemcpyHostToDevice),
                       "Failed to copy data from host to device");
        }

        /**
         * @brief Copy data from device(GPU) to host(CPU)
         * 
         * @param uiSize Size of data to copy (0: Copy all)
         */
        void download(unsigned int uiSize = 0)
        {
            // set size to copy all data
            if (uiSize == 0) uiSize = m_uiWidth * m_uiHeight * sizeof(T);

            // copy data from device to host
            assertCuda(cudaMemcpy(h_pData, d_pData, uiSize, cudaMemcpyDeviceToHost),
                       "Failed to copy data from device to host");
        }

        /**
         * @brief Get all data from host(CPU)
         * 
         * @return All data (1D array)
         */
        T* data()
        {
            // get all data from host
            return h_pData;
        }

        /**
         * @brief Get single data from host(CPU)
         * 
         * @param uiX X index
         * @param uiY Y index
         * @return Single data
         */
        T& at(unsigned int uiX, unsigned int uiY)
        {
            // get single data from host
            return h_pData[uiX + uiY * m_uiWidth];
        }

        /**
         * @brief Get single data from device(GPU)
         * @details Device(GPU) only
         * 
         * @param uiX X index
         * @param uiY Y index
         * @return Single data
         */
        __device__ T& operator()(unsigned int uiX, unsigned int uiY)
        {
            // get single data from device
            return d_pData[uiX + uiY * m_uiWidth];
        }

        /**
         * @brief Check if indices are inside of map
         * @details Device(GPU) only
         * 
         * @param uiX X index
         * @param uiY Y index
         * @param uiMargin Margin (More gap to check with)
         * @return Inside(true) or outside(false)
         */
        __device__ bool inside(unsigned int uiX, unsigned int uiY, unsigned int uiMargin = 0)
        {
            // check indices with margin
            return (uiX >= uiMargin && uiY >= uiMargin &&
                    uiX + uiMargin < m_uiWidth && uiY + uiMargin < m_uiHeight);
        }

        /**
         * @brief Check if indices are inside of map
         * @details Device(GPU) only @n Negative indices or margin return false
         * 
         * @param iX X index
         * @param iY Y index
         * @param iMargin Margin (More gap to check with)
         * @return Inside(true) or outside(false) 
         */
        __device__ bool inside(int iX, int iY, int iMargin = 0)
        {
            // check negative indices and margin
            if (iX < 0 || iY < 0 || iMargin < 0) return false;

            // check indices with margin by unsigned
            return inside((unsigned int)iX, (unsigned int)iY, (unsigned int)iMargin);
        }
    };
}
