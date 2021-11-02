#include "gauss.h"
#include "common.h"


namespace gauss {
/* ********************************* METHOD **********************************/
/**
 * The coefficients for the gauss kernel are:
 *   1/16  2/16  1/16
 *   2/16  4/16  2/16
 *   1/16  2/16  1/16
 * This corresponds to a standard deviation of 0.85
 *
 *****************************************************************************/
__device__ inline float size_3( const float*  f )
{
    return
            (f[0] + f[2] + f[6] + f[8] ) * 6.256912272348986e-02f +
            (f[1] + f[3] + f[5] + f[7] ) * 1.249999617975152e-01f +
            f[4]                                           * 2.497236619159801e-01f;
}




/* ********************************* METHOD **********************************/
/**
 * The coefficients for the gauss kernel are:
 *   a  b  c  b  a
 *   b  d  e  d  b
 *   c  e  f  e  c
 *   b  d  e  d  b
 *   a  b  c  b  a
 * with
 *   a = 2.969016743950497e-03f
 *   b = 1.330620989101365e-02f
 *   c = 2.193823127971464e-02f
 *   d = 5.963429543618012e-02f
 *   e = 9.832033134884574e-02f
 *   f = 1.621028216371266e-01f
 *
 * which corresponds to a standard deviation of 1.0
 *
 *****************************************************************************/
__device__ inline float size_5( const float*  f ) {

    return
            ( f[0] + f[4] + f[20] + f[24] )   * 2.969016743950497e-03f  +
            ( f[1]  + f[3]  + f[5]  + f[9] +
            f[15] + f[19] + f[21] + f[23] )   * 1.330620989101365e-02f  +
            ( f[2]  + f[10] + f[14] + f[22] ) * 2.193823127971464e-02f  +
            ( f[6]  + f[8]  + f[16] + f[18] ) * 5.963429543618012e-02f  +
            ( f[7]  + f[11] + f[13] + f[17] ) * 9.832033134884574e-02f  +
            f[12]                             * 1.621028216371266e-01f;
}


}
template<int delta
         /*1 => 3x3 kernel,
                   2=> 5x5 kernel*/>
__global__ void gauss_filter_kernel(
        const float*           input,
        float*           output,
        uint32_t    cols,
        uint32_t    rows,
        uint32_t    stride /* pixels, always in common?*/  )
{
    //TODO: consider replacing with generic conv

    // Current center pixel and center index
    auto col = blockIdx.x* blockDim.x  + threadIdx.x;
    auto row = blockIdx.y* blockDim.y  + threadIdx.y;
    auto index = col+ row* stride;
    if ( ( col >= cols ) || ( row >= rows ) ) return;
    if(col<delta||row<delta||col+delta>=cols||row+delta>=rows){
        output[index]=__ldg(&input[index]);
        return;
    }

    // Get pixel values
    float data[(delta*2+1)*(delta*2+1)];
    int i=0;
    for(int r=row-delta;r<row+delta+1;++r)
        for(int c=col-delta;c<col+delta+1;++c)
            data[i++]=__ldg(&input[c+ r* stride]);
    // consider replacing with designed filters
    // is there any meaningful reason to compute the value using anything other than float or double?
    // real variants are almost guaranteed to be problematic
    if constexpr (delta==1)
    {
        output[ index ] = gauss::size_3(data);
    }else{
        output[ index ] = gauss::size_5(data);
    }
}






bool
gauss_filter3(const Texture< float, true >&  input,
               Texture< float, true >&  output )
{
    mhere();
    output.resize(input);
    mhere();
    dim3 dimBlock( 64, 1, 1);
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );

    gauss_filter_kernel<1> <<< dimGrid, dimBlock, 0 >>>(input.cdata(),
                                                              output.data(),
                                                              input.cols(),
                                                              input.rows(),
                                                              output.stride() );
    mhere();
    return true;
}
bool
gauss_filter5(const Texture< float, true >&  input,
               Texture< float, true >&  output )
{
        mhere();
    output.resize(input);
        mhere();
    dim3 dimBlock( 64, 1, 1);
    dim3 dimGrid(  divUp( input.cols(),  dimBlock.x ),
                   divUp( input.rows(), dimBlock.y ),
                   1 );
    mhere();
    gauss_filter_kernel<2> <<< dimGrid, dimBlock, 0 >>>(input.cdata(),
                                                              output.data(),
                                                              input.cols(),
                                                              input.rows(),
                                                              output.stride() );
        mhere();
    return true;
}
