#pragma once
/* ********************************* FILE ************************************/
/** \file    cuda_helpers.h
 *
 * \brief    This header contains various helper functions that are essential when working with cuda
 *
 *
 * Cards I use:
 * - Laptop: gtx 750
 *  - Cuda Compute capability 5.0
 *  - cores 640
 *  - Shared Memory / SM		64 KB
 *  - Register File Size / SM	256 KB	256 KB
 *  - Active Blocks / SM		32
 *  - Memory Clock		5400 MHz
 *  - Memory Bandwidth		86.4 GB/s
 *  - L2 Cache Size		2048 KB
 *
 *
 *
 *
 * \remark
 * - c++11
 * - no dependencies
 * - header only
 * - cuda enabled
 * - tested by test_pose.cpp
 *
 *
 *  NOTE:
 *   - If any cuda kernel outputs nothing but otherwize seems to work, check if you are generating code for the appropriate compute cap
 *
 *
 * \todo
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2016-04-01
 * \note MIT licence
 *
 ******************************************************************************/

/**

  */




#include <assert.h>
#include <string>
#include <iostream>
#include <mlib/utils/cvl/matrix_adapter.h>


namespace cvl{
std::string getCudaErrorMsg(const cudaError_t& error);
/// checks if a cuda command did what it supposed to and converts the error code to human readable format
bool worked(const cudaError_t& error,std::string msg="");

template <class T>

/**
 * @brief cudaNew
 * @param elements
 * @return nullptr or the pointer to the allocated memory on device
 */
T* cudaNew(int elements)
{
    T* data=nullptr;
    cudaError_t error;
    // gives aligned memory.
    error = cudaMalloc((void **) &data, elements*sizeof(T));
    if (error != cudaSuccess)
    {
        std::cout<<"Failed to allocate memory on the device: size is: "<<sizeof(T)*elements/(1024.0*1024)<<"MB"<< "cuda error code is "<<(int)error<<" which means "<<cudaGetErrorString(error)<<std::endl;
        return nullptr;
    }
    return data;
}


namespace device
{

template <class T> T* allocate(int elements)
{
    T* data=nullptr;
    cudaError_t error;
    // gives aligned memory. probably... sigh...
    error = cudaMalloc((void **) &data, elements*sizeof(T));
    if (error != cudaSuccess)
    {
        std::cout<<"Failed to allocate memory on the device: size is: "<<sizeof(T)*elements/(1024.0*1024)<<"MB"<< "cuda error code is "<<(int)error<<" which means "<<getCudaErrorMsg(error)<<std::endl;
        return nullptr;
    }
    return data;
}


// deallocates on destruction
template<class T> struct Array
{
    //int device
    T* data=nullptr;
    int size=0;
    Array()=default;
    Array(T* ptr, int size):data(ptr),size(size){}
    ~Array() {if(data) cudaFree(data); size=0;}
    explicit operator bool() const { return data!=nullptr; }
};

}

/**
 * @brief devicePointer
 * @param p the pointer
 * @return if the pointer is __null or allocated on the device, since __null can be either or
 * &p[5] works aleast for the length of the allocated data
 */
bool devicePointer(const void* p);









template<class T>
/**
 * @brief copy from a arbitrary host or device pointer to the opposite, the to pointer which is allocated if it is nullptr
 * @param from
 * @param to
 * @param elements
 * copy from dev to host or vice versa automatically
 */
void copy(T*  from, T*& to, unsigned int elements){

    assert(from!=nullptr);// check sizes somehow?


    if(devicePointer(from)){
        //   std::cout<<"copy from dev: "<<elements<<std::endl;
        if(to==nullptr){

            to=new T[elements];
        }
        worked(cudaMemcpy(to,from, elements*sizeof(T), cudaMemcpyDeviceToHost));
    }else{
        // std::cout<<"copy to dev: "<<elements<<std::endl;
        if(to==nullptr)
            to=cudaNew<T>(elements);
        worked(cudaMemcpy(to, from, elements*sizeof(T), cudaMemcpyHostToDevice));
    }
    assert(to!=nullptr);
}



template<class T>
/**
 * @brief copy from a arbitrary host or device pointer to the opposite, the to pointer which is allocated if it is nullptr
 * @param from
 * @param to
 * @param elements
 * @param stream
 * copy from dev to host or vice versa automatically
 * \remark
 * - specifying the stream allows some async data transfer(2 at most for cuda compute 5)
 */
void copy(const T*  from, T*& to, unsigned int elements, cudaStream_t& stream){

    assert(from!=nullptr);// check sizes somehow?

    if(devicePointer(from)){
        //std::cout<<"copy from dev: "<<elements<<std::endl;
        if(to==nullptr)            to=new T[elements];
        bool good=worked(cudaMemcpyAsync(to,from, elements*sizeof(T), cudaMemcpyDeviceToHost, stream));
        assert(good && "cuda copy problem");
    }else{
        //std::cout<<"copy to dev: "<<elements<<std::endl;
        if(to==nullptr)            to=cudaNew<T>(elements);
        bool good=worked(cudaMemcpyAsync(to,from, elements*sizeof(T), cudaMemcpyHostToDevice, stream));
        assert(good && "cuda copy problem");
    }
    assert(to!=nullptr);
}


std::string getDeviceData();














}
