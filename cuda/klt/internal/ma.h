#pragma once
#include <cuda_runtime.h>
#include <cstdint>
template<class T>
__device__
inline T& at(T* data, int stride, int row, int col){return data[stride*row + col];}

template<class T>
__device__
inline const T at(const T* data, int stride, int row, int col){return __ldg(&data[stride*row + col]);}
template<class T>
__device__
inline const T load(const T* data, int stride, int row, int col){return __ldg(&data[stride*row + col]);}

