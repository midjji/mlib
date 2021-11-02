#include "klt/internal/non_maxima_supression.h"
#include <mlib/utils/mlibtime.h>
#include "common.h"



__global__ void non_max_kernel_exact(
        float* output,
        const float* input,
        float    radius,
        int32_t    rows,
        int32_t    cols,
        int32_t    stride )
{
    //if(radius>3) printf("non_max_kernel_exact: This is very slow %f, ", radius);
    // start time was 2 ms, but at what radius?
    // Current center pixel
    int32_t col        =  blockIdx.x* blockDim.x  + threadIdx.x;
    int32_t row        =  blockIdx.y* blockDim.y  + threadIdx.y;


    // in image,
    if(row >= rows) return;
    if(col >= cols ) return;

    // The reference value
    float        center = __ldg(&input[row*stride+col]);
    if(center==0.0f)
    {
        output[row*stride+col] = center;
        return;
    }
    float res=center;



    int32_t row0=row - radius;if(row0<0) row0=0;
    int32_t row1=row+ radius+1.5f; if(row1>=rows) row1=rows-1;
    int32_t col0=col - radius; if(col0<0) col0=0;
    int32_t col1=col + radius+1.5f;if(col1>=cols) col1=cols-1;

    // the test order is wrong
    for ( int32_t r = row0;          r<row1 && res>0.0f; ++r )
    {
        for ( int32_t c = col0;          c<col1; ++c )
        {

            if(r==0 && c==0) continue;

            float dr=(row-r);dr*=dr;
            float dc=(col-c);dc*=dc;
            if(dr+dc>radius*radius) continue;
            float value = __ldg(&input[r*stride+c]);

            if ( value <= center ) continue;
            res=0.0f;
            break;
        }
    }
    // Set output pixel to 0.0f
    output[row*stride+col] = res;
    return;
}







void non_max_supression(
        const  Texture< float, true>&  input,
        Texture< float, true>&  output,
        float radius)
{
    output.resize(input);




    dim3 dimBlock( 64,1, 1 );

    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1);

    non_max_kernel_exact<<< dimGrid, dimBlock, 0 >>>(
                                                 output.data(),
                                                 input.cdata(),
                                                 radius,
                                                 input.rows(),
                                                 input.cols(),
                                                 input.stride() );
mhere();
}



