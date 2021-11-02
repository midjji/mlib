#include "ma.h"
#include "harris.h"
#include "common.h"
namespace klt {





namespace kernel{
__global__
/**
 * @brief harris_corner_score
 * @param output
 * @param dx
 * @param dy
 * @param half_window
 * @param cols
 * @param rows
 *
 * Computes the structure tensor then finds the minimum eigenvalue.
 * its essentially how 2d is the point.
 * This requires a local window, but that window can be very small, e.g. 3x3 or less, just not 1x1. making 3x3 the smallest sensible value.
 * The klt can sortof track any patch which within its window is locally 2d, meaning matching patch sizes could make sense.
 * but the patches that catches are the ones were one part of the patch has a line,
 * and a separate part has a different line. This does not quite make sense to use as a feature, since they seem more likely to be unrelated.
 * What might make sense is to require features to be trackable in multiple pyramid levels, though.
 *
 * Thus larger windows do not really make sense, nor do they seem to improve performance.
 * The cornerness score is non sensible, minimum is 0, local maximas are sensible.
 * tough they should be evaluated on a lopassed gradient, which since the gradient is unknown and only the lowpassed gradient is available is fine.
 * The score could be subpixel, but there is little reason to assume that is better
 *
 * In practice almost everything could be used as a trackable patch except the zero ones. So dont worry so much.
 * There is definetly place for a different kind of feature to be tracked entirely too though.
 *
 * This is the exact one, instead of one of the slightly faster proxies,
 * all the time will be spent reading gradients anyways...
 */
void harris_corner_score( float* output,
                                 const float* dx,
                                 const float* dy,
                                 int32_t    rows,
                                 int32_t    cols,
                                 int32_t stride)
{
    // Current center pixel
    int32_t col = ( blockIdx.x* blockDim.x ) + threadIdx.x;
    int32_t row = ( blockIdx.y* blockDim.y ) + threadIdx.y;
    if(rows<=row) return;
    if(cols<=col) return;


    constexpr int32_t half_window=3; // 3x3 its enough to resolve the lowpass req issue,
    if(
            (col-half_window<0)||
            (cols<=col+half_window+1)||
            (row-half_window<0)||
            rows<=row+half_window+1)
    {

        at(output, stride, row, col)= 0; // lowest possible
        return;
    }



    // sum over local gradients, the reason for this is deep and unclear,
    // marias argument is unsatisfactory
    // perhaps precompute the e_min at the same time as computing the gradients?
    float xx = 0.0f;
    float xy = 0.0f;
    float yy = 0.0f;

    for ( int32_t r = row -half_window; r < row + half_window+1; ++r )
        for ( int32_t c = col -half_window; c < col + half_window+1; ++c )
        {
            float x = at(dx, stride, r, c);
                      xx += ( x* x );
            float y = at(dy, stride, r, c );
            xy += ( x* y );
            yy += ( y* y );
        }


    // exact minimum eigenvalue computation,
    /*
        (a,b
         b,c)
         gives eigenvalues:
         (a-e)*(c-e) -b*b =0
         ac + e^2 -(a + c)e -b*b =0
         e^2 + pe + q =0, for
         p= -a -c
         q= ac - b^2
         and therefore
         e = -0.5*p \pm sqrt(p*p -4q)*0.5
         simplifying:
         v= p*p -4q = (a - c)^2 +4b^2 >=0
         e_min = 0.5(-p - sqrt(v))
         e_min = 0.5(a + c - sqrt(v)); // we are computing a cornerness score, not a eigenvalue... skip the 0.5
         */
    //float a= xx;
    //float b= xy;
    //float c= yy;
    float v= (xx-yy)*(xx-yy) + 4.0f*xy*xy;
    at(output, stride, row, col) =  (xx + yy - sqrtf(v));

}

}

void harris_corner_score(
        const Texture< float, true>&  dx,
        const Texture< float, true>&  dy,
        Texture< float, true>&  output, int stream)
{
    //lightning fast, 0.1ms
    // Calculate minimum eigenvalues
    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid(  divUp( output.cols(),  dimBlock.x ),
                   divUp( output.rows(), dimBlock.y ),
                   1 );


    if(output.rows()!=dx.rows())
        std::cout<<"output dx row diff: "<<output.rows()<<" "<<dx.rows()<<std::endl;
    if(output.cols()!=dx.cols())
        std::cout<<"bad input to harris, cols: "<<output.rows()<<" "<<dx.rows()<<std::endl;
    if(output.rows()!=dy.rows())
        std::cout<<"bad input to harris, dy rows: "<<output.rows()<<" "<<dx.rows()<<std::endl;
    if(output.cols()!=dy.cols())
        std::cout<<"bad input to harris, dy cols: "<<output.rows()<<" "<<dx.rows()<<std::endl;

    if(output.stride()!=dx.stride())
        std::cout<<"bad input to harris, stride"<<output.stride()<<" "<<dx.stride()<<std::endl;



    kernel::harris_corner_score<<< dimGrid, dimBlock, stream >>>(output.data(),
                                                             dx.cdata(),
                                                             dy.cdata(),
                                                             output.rows(),
                                                             output.cols(),
                                                             output.stride());
}
}
