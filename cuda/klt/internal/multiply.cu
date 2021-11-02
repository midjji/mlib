
#include "multiply.h"


#include "klt/internal/texture.h"
#include "common.h"

__global__ void
set_kernel(float* input,
          uint32_t    cols,
          uint32_t    rows,
          uint32_t    stride,
                float value)
{
    uint32_t col =  blockIdx.x* blockDim.x  + threadIdx.x;
    uint32_t row =  blockIdx.y* blockDim.y  + threadIdx.y;
    if( col >= cols ) return;
    if( row >= rows ) return;
    input[ col+ row*stride ] = value;
}


bool set2( Texture< float, true >&  input,
                     float value )
{
    if(input.rows()*input.cols()==0) return true;
    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );
    set_kernel <<< dimGrid, dimBlock, 0 >>>(input.data(),
                                          input.cols(),
                                          input.rows(),
                                          input.stride(),
                                            value);
    return true;
}
bool set2( Texture< float, false >&  image, float val){
    for(int i=0;i<image.rows()*image.stride();++i)
        image.data()[i]=val;
    return true;
}

__global__ void
multiply_kernel( const float* input,
          const float* factor,
          float* output,
          uint32_t    cols,
          uint32_t    rows,
          uint32_t    stride )
{
    // the images must share stride and size
    uint32_t col =  blockIdx.x* blockDim.x  + threadIdx.x;
    uint32_t row =  blockIdx.y* blockDim.y  + threadIdx.y;
    if( col >= cols ) return;
    if( row >= rows ) return;
    uint32_t  idx_i    =  col+ row*stride;
    output[ idx_i ] = __ldg(&input[  idx_i ]) * __ldg(&factor[ idx_i ]);
}



void multiply( const Texture< float, true>&  input,
               const   Texture< float, true>&  factor,
                 Texture< float, true>&  output )
{

    require(input.same_size_and_stride(factor) && input.same_size_and_stride(output),"");

    // Multiply images
    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );

    multiply_kernel <<< dimGrid, dimBlock, 0 >>>(input.cdata(),
                                          factor.cdata(),
                                          output.data(),
                                          input.cols(),
                                          input.rows(),
                                          input.stride());
}


