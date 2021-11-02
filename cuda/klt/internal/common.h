#pragma once
#include <cstdint>
#include <cuda_runtime.h>


inline int divUp( int a, int b )
{
    return (a+b-1)/b;
}
