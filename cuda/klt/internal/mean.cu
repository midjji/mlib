
#include "mean.h"
#include "common.h"


__global__ void
mean_estimate(const float*        input,
                float*        output,
                                 const int32_t cols,
                 const int32_t rows,
                 const int32_t stride,

                 const int32_t f_halfWinWidth_i,
                 const int32_t f_halfWinHeight_i )
{
    // mean conv, or mean estimate?
    // Current center pixel and center index
    auto col = blockIdx.x* blockDim.x  + threadIdx.x;
    auto row = blockIdx.y* blockDim.y  + threadIdx.y;
    auto index = col+ row* stride;
    if ( ( col >= cols ) || ( row >= rows ) ) return;


    // Get pixel values
    float v=0.0f;
    int i=0;
    for(int r=row-f_halfWinHeight_i;r<row+f_halfWinHeight_i+1;++r)
        for(int c=col-f_halfWinWidth_i;c<col+f_halfWinWidth_i+1;++c)
        {
            if( c<0|| r<0) continue;
            if(c>=cols|| r>=rows) continue;

            v+=__ldg(&input[stride*r + c]);
            //v += tex2D( g_meanInputTex_32f, c, r );
            ++i;
        }
    v/=float(i);
    output[ index ] = v;

}



bool estimate_mean( Texture< float, true >& input,
                    Texture< float, true >&       output,
                    int half_width,
                    int half_height )
{
    // Ensure output image has same dimensions as the input image.
    output.resize( input );


    // Call the kernel
    dim3 dimBlock( 16, 8, 1 );
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );

    mean_estimate<<< dimGrid, dimBlock, 0 >>>(input.data(),
                                                 output.data(),
                                                   input.cols(),
                                                   input.rows(),
                                                   input.stride(),
                                                   half_width,
                                                   half_height );

    return true;
}

