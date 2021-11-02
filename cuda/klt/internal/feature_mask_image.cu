#include "ma.h"
#include "feature_mask_image.h"
#include "klt/internal/multiply.h"
#include "common.h"

#include <iostream>
#include <mlib/utils/mlibtime.h>
mlib::Timer timer=mlib::Timer("feature mask");

namespace kernel {

namespace {
__device__ inline int32_t cap(int32_t a, int32_t l, int32_t h){
    if(a<l) return l;
    if(a>h) return h;
    return a;
}
}

__global__ void feature_mask( float* mask,
                              int32_t    rows,
                              int32_t    cols,
                              int32_t    stride,
                              const float* features,
                              int32_t num_features,
                              float    radius )
{

    int32_t fid        =  blockIdx.x* blockDim.x  + threadIdx.x;
    if(fid>=num_features) return;
    float row=__ldg(&features[fid*2]);
    float col=__ldg(&features[fid*2+1]);

    int32_t row0=row - radius;
    int32_t row1=row+ radius+1.5f; // round up
    int32_t col0=col - radius;
    int32_t col1=col + radius+1.5f;
    row0=cap(row0, 0,rows-1);
    col0=cap(col0, 0,cols-1);
    row1=cap(row1, 0,rows-1);
    col1=cap(col1, 0,cols-1);

    for ( int32_t r = row0;          r<row1; ++r )
    {
        for ( int32_t c = col0;          c<col1; ++c )
        {
            float dr=(row-r);dr*=dr;
            float dc=(col-c);dc*=dc;
            if(dr+dc>radius*radius) continue;
            mask[r*stride+c] = 0.0f;
        }
    }
}
}
void feature_mask(Texture< float,true >&  mask,
                  const Texture< float, true>& features,
                  float radius)
{

    set2(mask,1);

    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(  divUp( features.elements()/2,dimBlock.x),1,
                   1 );
    kernel::feature_mask<<< dimGrid, dimBlock, 0 >>>(mask.data(),
                                                     mask.rows(),
                                                     mask.cols(),
                                                     mask.stride(),
                                                     features.cdata(),
                                                     features.elements()/2, radius);
}

