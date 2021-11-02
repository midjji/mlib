#include <iostream>


#include "klt/internal/convert.h"
#include "common.h"

namespace kernel
{

template <class A, class B>
__global__ void
ConvertKernel( const A*       input,
               B*         output,
               const uint32_t cols,
               const uint32_t rows,
               const uint32_t stride )
{
    // Current center pixel
    const int32_t x           = blockIdx.x*blockDim.x  + threadIdx.x;
    const int32_t y           = blockIdx.y*blockDim.y  + threadIdx.y;


    // Check for valid input range
    if ( ( x < cols ) &&
         ( y < rows ) )
    {

        output[ stride*y +x ] =B( __ldg(&input[ stride*y +x] ));
    }
}
}
template < class A, class B >
void
convertAB( const Texture< A, true >&  input,
              Texture< B, true >&  output )
{
    output.resize( input );
    // Execute kernel
    dim3 dimBlock( 256, 1, 1); // num bytes aligned
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );
    kernel::ConvertKernel<A,B><<< dimGrid, dimBlock, 0 >>>(input.cdata(),
                                                     output.data(),
                                                     output.cols(),
                                                     output.rows(),
                                                     input.stride() );




}

void convert(
        const Texture< std::uint16_t, true>&  input,
        Texture< float, true >&  output){
    return convertAB(input,output);
}


void convert(
        const Texture< std::uint8_t, true>&  input,
        Texture< std::uint16_t, true >&  output){
    return convertAB(input,output);
}
void convert(
        const Texture< std::uint8_t, true>&  input,
        Texture< float, true >&  output){
    return convertAB(input,output);
}
