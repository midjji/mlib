#include "downsample.h"
#include "common.h"


namespace  {



__device__
inline float size_5_vals(int val)
{
    switch(val){
    case 0: return 2.969016743950497e-03f;
    case 4: return 2.969016743950497e-03f;
    case 20: return 2.969016743950497e-03f;
    case 24: return 2.969016743950497e-03f;

    case 1: return 1.330620989101365e-02f;
    case 3: return 1.330620989101365e-02f;
    case 5: return 1.330620989101365e-02f;
    case 9: return 1.330620989101365e-02f;
    case 15: return 1.330620989101365e-02f;
    case 19: return 1.330620989101365e-02f;
    case 21: return 1.330620989101365e-02f;
    case 23: return 1.330620989101365e-02f;

    case 2: return 2.193823127971464e-02f;
    case 10: return 2.193823127971464e-02f;
    case 14: return 2.193823127971464e-02f;
    case 22: return 2.193823127971464e-02f;
    case 6: return 5.963429543618012e-02f;
    case 8: return 5.963429543618012e-02f;
    case 16: return 5.963429543618012e-02f;
    case 18: return 5.963429543618012e-02f;
    case 7: return 9.832033134884574e-02f;
    case 11: return 9.832033134884574e-02f;
    case 13: return 9.832033134884574e-02f;
    case 17: return 9.832033134884574e-02f;
    case 12: return 1.621028216371266e-01f;
    default: return 0;
    }
}


__device__
inline float size_3_vals(int val){
    switch(val){
    case 0: return 6.256912272348986e-02f;
    case 1: return 1.249999617975152e-01f;
    case 2: return 6.256912272348986e-02f;
    case 3: return 1.249999617975152e-01f;
    case 4: return 2.497236619159801e-01f;
    case 5: return 1.249999617975152e-01f;
    case 6: return 6.256912272348986e-02f;
    case 7: return 1.249999617975152e-01f;
    case 8: return 6.256912272348986e-02f;
    default: return 0;
    }
}

}
namespace kernel {

template<int delta>
__global__
void
downsample_gauss_filter(const float*           input,
                          uint32_t input_rows,
                          uint32_t input_cols,
                          uint32_t input_stride,
                          float*           output,
                          uint32_t output_rows,
                          uint32_t output_cols,
                          uint32_t output_stride,
                          float scale  // the scaling of the values, not the image
                          )
{


    // Current pixel in the downsampled image (upper image in the pyramid)
    const int32_t ocol = blockIdx.x* blockDim.x  + threadIdx.x;
    const int32_t orow = blockIdx.y* blockDim.y  + threadIdx.y;

    // If center pixel is out of image, abort
    if ( ( ocol >= output_cols ) || ( orow >= output_rows ) )
        return;

    // Center pixel location in the original image (lower image in the pyramid).
    // Coordinate transformation: x_low = 2*(x_up + 1) - 1 = 2*x_up + 1
    // Note: x_low/y_low are always greater than 0!
    const int32_t icol    = ocol* 2  + 1;
    const int32_t irow    = orow* 2  + 1;

    int i=0;
    float v;
    float tot=0;
    for(int r=irow-delta;r<irow+delta+1;++r)
        for(int c=icol-delta;c<icol+delta+1;++c)
        {

            if(r<0||c<0||c>=input_cols||r>=input_rows)
                v=__ldg(&input[icol+ irow* input_stride]);
            else
                v=__ldg(&input[input_stride*r + c]);
            //    data[i++]=v;
            if constexpr (delta==1)
                    tot+=v*size_5_vals(i++);
            else
                    tot+=v*size_3_vals(i++);
        }
    output[ocol + orow*output_stride] = tot*scale;
}


}



bool
downsample_gauss_filter3( const Texture< float, true >& input,
                          Texture< float, true >& output, float scale )
{
    mhere();
    output.resize_rc(input.rows()/2,input.cols()/2);
    mhere();

    dim3 dimBlock( 16, 16, 1);


    dim3 dimGrid(  divUp( output.cols(),  dimBlock.x ),
                   divUp( output.rows(), dimBlock.y ),
                   1 );
    mhere();
    kernel::downsample_gauss_filter<1> <<< dimGrid, dimBlock, 0 >>>( input.cdata(),
                                                                    input.rows(),
                                                                    input.cols(),
                                                                    input.stride(),
                                                                    output.data(),
                                                                    output.rows(),
                                                                    output.cols(),
                                                                    output.stride(),scale );
    mhere();

    return true;
}




bool
downsample_gauss_filter5( const Texture< float, true >& input,
                          Texture< float, true >& output, float scale )
{


    output.resize_rc(input.rows()/2,input.cols()/2);
    dim3 dimBlock(16, 16, 1);


    dim3 dimGrid(  divUp( output.cols(),  dimBlock.x ),
                   divUp( output.rows(), dimBlock.y ),
                   1 );

    kernel::downsample_gauss_filter<2>
            <<< dimGrid, dimBlock, 0 >>>(
                                           input.cdata(),
                                           input.rows(),
                                           input.cols(),
                                           input.stride(),
                                           output.data(),
                                           output.rows(),
                                           output.cols(),
                                           output.stride(),scale );

    return true;
}





__global__ void
downsample_max_kernel(
        const float*           input,
        uint32_t input_rows,
        uint32_t input_cols,
        uint32_t input_stride,
        float*           output,
        uint32_t output_rows,
        uint32_t output_cols,
        uint32_t output_stride)
{


    const int32_t ocol = blockIdx.x* blockDim.x  + threadIdx.x;
    const int32_t orow = blockIdx.y* blockDim.y  + threadIdx.y;

    if ( ( ocol >= output_cols ) || ( orow>= output_rows ) )
        return;
    /*
    if(orow*2>= input_rows||
       ocol*2>= input_cols){

        output[orow*output_stride + ocol] = input[(input_rows-1)*]
    }
    */

    float a=__ldg(&input[input_stride*orow*2 + ocol*2]);
    float b=__ldg(&input[input_stride*orow*2 + ocol*2 +1]);
    if(a<b)a=b;
    float c=__ldg(&input[input_stride*(orow*2+1) + ocol*2]);
    float d=__ldg(&input[input_stride*(orow*2+1) + ocol*2 +1]);
    if(c<d)c=d;
    if(a<c)a=c;
    output[output_stride*orow + ocol]=a;
}


void
downsample_max( const Texture< float, true >& input,
                Texture< float, true >& output)
{


    output.resize_rc(input.rows()/2,input.cols()/2);
    dim3 dimBlock(64, 16, 1);


    dim3 dimGrid(  divUp( output.cols(),  dimBlock.x ),
                   divUp( output.rows(), dimBlock.y ),
                   1 );

    downsample_max_kernel
            <<< dimGrid, dimBlock, 0 >>>(
                                           input.cdata(),
                                           input.rows(),
                                           input.cols(),
                                           input.stride(),
                                           output.data(),
                                           output.rows(),
                                           output.cols(),
                                           output.stride() );


}
