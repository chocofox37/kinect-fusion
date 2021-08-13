#pragma once

#include "config.cuh"
#include "helper.cuh"

namespace kf
{
    /**
     * @brief 2D array that saves data between host(CPU) & device(GPU)
     * 
     * @tparam T Type of data to save for each pixel
     */
    template<typename T>
    class Map
    {
    private:

        unsigned int m_uiWidth;  /**< Width (number of pixels in X direction) */
        unsigned int m_uiHeight; /**< Height (number of pixels in Y direction) */
        dim3 m_dimBlock; /**< Dimension of block to run CUDA kernel function */
        dim3 m_dimGrid;  /**< Dimension of grid to run CUDA kernel function */
        T* h_pData; /**< Pointer of data from host(CPU) */
        T* d_pData; /**< Pointer of data from device(GPU) */

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
         * @details Be used to run CUDA kernel function
         */
        const dim3& block = m_dimBlock;

        /**
         * @brief Dimension of grid
         * @details Be used to run CUDA kernel function
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
         * @param uiWidth Width (number of pixels in X direction)
         * @param uiHeight Height (number of pixels in Y direction)
         */
        Map(unsigned int uiWidth, unsigned int uiHeight) :
            m_uiWidth(uiWidth),
            m_uiHeight(uiHeight),
            m_dimBlock(KF_MAP_BLOCK_SIZE_X, KF_MAP_BLOCK_SIZE_Y),
            m_dimGrid((uiWidth  + KF_MAP_BLOCK_SIZE_X - 1) / KF_MAP_BLOCK_SIZE_X,
                      (uiHeight + KF_MAP_BLOCK_SIZE_Y - 1) / KF_MAP_BLOCK_SIZE_Y),
            h_pData(nullptr),
            d_pData(nullptr)
        {
            // do nothing
        }

        /**
         * @brief Allocate host(CPU) & device(GPU) memory
         * @details Must call once before using this class (memory allocation)
         */
        void allocate()
        {
            // check memory allocation state
            ASSERT_CXX(!h_pData && !d_pData, "Failed to allocate memory (already allocated)")

            // allocate host and device memory
            ASSERT_CXX(h_pData = new T[m_uiWidth * m_uiHeight],
                       "Failed to allocate host memory");
            ASSERT_CUDA(cudaMalloc((void**)&d_pData, m_uiWidth * m_uiHeight * sizeof(T)),
                        "Failed to allocate device memory");
        }

        /**
         * @brief Free host(CPU) & device(GPU) memory
         * @details Must call once after using this class (memory leak)
         */
        void free()
        {
            // check memory allocation state
            ASSERT_CXX(h_pData && d_pData, "Failed to free memory (never allocated)")

            // free host and device memory
            delete[] h_pData;
            ASSERT_CUDA(cudaFree(d_pData), "Failed to free device memory");
        }

        /**
         * @brief Copy data from host(CPU) to device(GPU)
         * 
         * @param uiSize Size of data to copy (0: copy all)
         */
        void upload(unsigned int uiSize = 0)
        {
            // set size to copy all data
            if (uiSize == 0)
                uiSize = m_uiWidth * m_uiHeight * sizeof(T);

            // copy data from host to device
            ASSERT_CUDA(cudaMemcpy(d_pData, h_pData, uiSize, cudaMemcpyHostToDevice),
                        "Failed to copy data from host to device");
        }

        /**
         * @brief Copy data from device(GPU) to host(CPU)
         * 
         * @param uiSize Size of data to copy (0: copy all)
         */
        void download(unsigned int uiSize = 0)
        {
            // set size to copy all data
            if (uiSize == 0)
                uiSize = m_uiWidth * m_uiHeight * sizeof(T);

            // copy data from device to host
            ASSERT_CUDA(cudaMemcpy(h_pData, d_pData, uiSize, cudaMemcpyDeviceToHost),
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
         * @param iX X index
         * @param iY Y index
         * @param uiMargin Margin (more gap to check with)
         * @return Inside(true) or outside(false) 
         */
        __device__ bool inside(int iX, int iY, unsigned int uiMargin = 0) const
        {
            // check indices with margin
            return (iX - (int)uiMargin >= 0 && iX + (int)uiMargin < (int)m_uiWidth && 
                    iY - (int)uiMargin >= 0 && iY + (int)uiMargin < (int)m_uiHeight);
        }
    };
}
