#include "ma.h"

#include "scharr.h"
#include "common.h"
/**
 * Calculate gradient in x- and y-direction using a 3x3 Scharr kernel.
 * consider the separable variant...
 * Also since this is to be used as actual gradient, consider fixing the scaling so it is, you know, the actual gradient...
 *
 */
__global__ void
scharr_dx_dy_kernel( float*  dx,
                           float*  dy,
                           const float* image,
                           int     rows,
                           int     cols,
                           int     stride )
{
    // Current center pixel
    int col           = ( blockIdx.x* blockDim.x ) + threadIdx.x;
    int row           = ( blockIdx.y* blockDim.y ) + threadIdx.y;

    // not in the image
    if(row>=rows|| col>=cols) return;


    // filter size is 3x3, centered, so
    if(row==0||col==0||row+1>=rows||col+1>=cols)
    {
        at(dx, stride, row, col)=0;
        at(dy, stride, row, col)=0;
        return;
    }

    /* consider sep conv
    * Scharr kernel for dx value:
    *  -0.09375  0  0.09375
    *  -0.3125   0  0.3125
    *  -0.09375  0  0.09375
    *
    *
    * Scharr kernel for dy value:
    *  -0.09375  -0.3125  -0.09375
    *   0         0        0
    *   0.09375   0.3125   0.09375
      */

    // minimize read count...
    float dx_=0;
    float dy_=0;
    float a=load(image, stride,  row - 1, col - 1 );
    dx_+=-a*0.09375f;
    dy_+=-a*0.09375f;
    a=load(image, stride,  row - 1, col );
    dy_+=-a*0.3125f;

    a=load(image, stride,  row - 1, col +1);
    dx_+=a*0.09375f;
    dy_+=-a*0.09375f;

    a=load(image, stride,  row, col -1);
    dx_+=-0.3125f*a;
    a=load(image, stride,  row, col +1);
    dx_+= 0.3125f*a;

    a=load(image, stride,  row+1, col -1);
    dx_+=-a*0.09375f;
    dy_+= a*0.09375f;

    a=load(image, stride,  row+1, col);
    dy_+=a*0.3125f;

    a=load(image, stride,  row+1, col+1);
    dx_+=a*0.09375f;
    dy_+=a*0.09375f;

    int index = ( row* stride ) + col;
    dx[ index ] = dx_;
    dy[ index ] = dy_;
}



void
scharr_dx_dy(    const    Texture< float, true >&  image,
                                     Texture< float, true >&  dx,
                                     Texture< float, true >&  dy )
{

    dx.resize(image);
    dy.resize(image);
    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid(  divUp( image.cols(),  dimBlock.x ),
                   divUp( image.rows(),  dimBlock.y ),
                   1 );

    scharr_dx_dy_kernel <<< dimGrid, dimBlock, 0 >>>( dx.data(),
                                                            dy.data(),
                                                            image.cdata(),
                                                            dx.rows(),
                                                            dx.cols(),
                                                            dx.stride() );
    mhere();


}
