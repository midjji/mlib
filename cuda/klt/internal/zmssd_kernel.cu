

#include "klt/internal/common.h"
#include "klt/internal/klt_kernels.h"
#include <mlib/utils/mlog/log.h>
namespace klt{



/*
 * Calculate the sum of the gradient matrix components.
 * The start position is the first pixel to use, the last pixel is
 *  f_startX_f + f_winWidth_i  - 1,
 *  f_startY_f + f_winHeight_i - 1
 *
 */
__device__ void
CalculateGradientSumsTex_ZMSSD_32f(
        const cudaTextureObject_t dx,
        const cudaTextureObject_t dy,
        const cudaTextureObject_t previous,
        float           f_startX_f,
        float           f_startY_f,
        int             f_winWidth_i,
        int             f_winHeight_i,
        float&  f_sumIxx_f,
        float&  f_sumIxy_f,
        float&  f_sumIyy_f,
        float&          fr_mean_f )
{
    f_sumIxx_f = 0.0f;
    f_sumIxy_f = 0.0f;
    f_sumIyy_f = 0.0f;
    fr_mean_f  = 0.0f;

    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float y_f = f_startY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float x_f = f_startX_f + float(x);

            const float ix_f = tex2D<float>( dx, x_f, y_f );
            const float iy_f = tex2D<float>( dy, x_f, y_f );

            f_sumIxx_f += ( ix_f* ix_f );
            f_sumIxy_f += ( ix_f* iy_f );
            f_sumIyy_f += ( iy_f* iy_f );
            fr_mean_f  += tex2D<float>( previous, x_f, y_f );
        }
    }

    fr_mean_f /= float( f_winWidth_i * f_winHeight_i );
}


__device__ void
CalculateBVectorTex_ZMSSD_32f(
        const cudaTextureObject_t dx,
        const cudaTextureObject_t dy,
        const cudaTextureObject_t previous,
        const cudaTextureObject_t current,
        float       f_startLastX_f,
        float       f_startLastY_f,

        float       f_startCurrentX_f,
        float       f_startCurrentY_f,
        int         f_winWidth_i,
        int         f_winHeight_i,
        float&  f_bx_f,
        float&  f_by_f,
        float       f_meanLast_f )
{
    f_bx_f = 0.0f;
    f_by_f = 0.0f;


    // Calculate mean of current image patch
    float meanCurr_f = 0.0;

    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float currY_f = f_startCurrentY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float currX_f = f_startCurrentX_f + float(x);

            meanCurr_f   += tex2D<float>( current, currX_f, currY_f );
        }
    }

    meanCurr_f /= float( f_winWidth_i * f_winHeight_i );


    // Calculate B-Vector components
    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float lastY_f = f_startLastY_f    + float(y);
        const float currY_f = f_startCurrentY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float lastX_f = f_startLastX_f    + float(x);
            const float currX_f = f_startCurrentX_f + float(x);

            const float ix_f = tex2D<float>( dx, lastX_f, lastY_f );
            const float iy_f = tex2D<float>( dy, lastX_f, lastY_f );
            const float dI_f = tex2D<float>( previous, lastX_f, lastY_f ) -
                    tex2D<float>( current, currX_f, currY_f )  +
                    ( meanCurr_f - f_meanLast_f );

            f_bx_f     += ( dI_f* ix_f );
            f_by_f     += ( dI_f* iy_f );
        }
    }
}


__device__ float
CalculateSADTex_ZMSSD_32f(      const cudaTextureObject_t previous,
                                const cudaTextureObject_t current,
                                float       f_startLastX_f,
                                float       f_startLastY_f,
                                float       f_startCurrentX_f,
                                float       f_startCurrentY_f,
                                int         f_winWidth_i,
                                int         f_winHeight_i )
{
    float sad_f = 0.0f;

    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float lastY_f = f_startLastY_f    + float(y);
        const float currY_f = f_startCurrentY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float lastX_f = f_startLastX_f    + float(x);
            const float currX_f = f_startCurrentX_f + float(x);

            const float dI_f = tex2D<float>( previous, lastX_f, lastY_f ) -
                    tex2D<float>( current, currX_f, currY_f );

            sad_f      += fabsf( dI_f );
        }
    }

    return sad_f;
}


__device__ float
CalculateZSADTex_ZMSSD_32f( const cudaTextureObject_t dx,
                            const cudaTextureObject_t dy,
                            const cudaTextureObject_t previous,
                            const cudaTextureObject_t current,
                            float       f_startLastX_f,
                            float       f_startLastY_f,
                            float       f_startCurrentX_f,
                            float       f_startCurrentY_f,
                            int         f_winWidth_i,
                            int         f_winHeight_i )
{
    float meanLast_f = 0.0f;
    float meanCurr_f = 0.0f;
    
    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float lastY_f = f_startLastY_f    + float(y);
        const float currY_f = f_startCurrentY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float lastX_f = f_startLastX_f    + float(x);
            const float currX_f = f_startCurrentX_f + float(x);

            meanLast_f += tex2D<float>( previous, lastX_f, lastY_f );
            meanCurr_f += tex2D<float>( current, currX_f, currY_f );
        }
    }
    
    meanLast_f /= float( f_winHeight_i * f_winWidth_i );
    meanCurr_f /= float( f_winHeight_i * f_winWidth_i );

    
    float zsad_f = 0.0f;

    for ( int y=0; y < f_winHeight_i; ++y )
    {
        const float lastY_f = f_startLastY_f    + float(y);
        const float currY_f = f_startCurrentY_f + float(y);

        for ( int x=0; x < f_winWidth_i; ++x )
        {
            const float lastX_f = f_startLastX_f    + float(x);
            const float currX_f = f_startCurrentX_f + float(x);

            const float dI_f = ( ( tex2D<float>( previous, lastX_f, lastY_f ) - meanLast_f ) -
                                 ( tex2D<float>( current, currX_f, currY_f ) - meanCurr_f ) );

            zsad_f      += fabsf( dI_f );
        }
    }

    return zsad_f;
}


__global__ void
CalculateKLT_ZMSSD(      const cudaTextureObject_t previous,
                         const cudaTextureObject_t dx,
                         const cudaTextureObject_t dy,                         
                         const cudaTextureObject_t current,
                         SCUDAKLTFeature_t*  features,
                         int                 num_features,
                         int                 half_window,
                         int                 num_iterations,
                         float               min_displacement,
                         float               max_displacement,
                         int                 level,
                         int                 rows,
                         int                cols)
{
    const unsigned int index = ( blockIdx.x* blockDim.x ) + threadIdx.x;

    // Abort if index is not valid
    if ( index >= num_features )
        return;


    // Get feature data in current pyramid level
    float prev_col_, prev_row_, current_col_, current_row_;

    // Feature to work with
    SCUDAKLTFeature_t  feature = features[ index ];

    // Abort if feature is not tracked (any more)
    if ( feature.status_i != 0 )
        return;


    // Transform previous feature position into current pyramid level.
    const float scale = 1.0f / float( 1 << level );


    prev_col_    = ( feature.prev_col_* scale );
    prev_row_    = ( feature.prev_row_* scale );
    current_col_ = ( feature.current_col_* scale );
    current_row_ = ( feature.current_row_* scale );




    // Calculate start position of previous image window
    const float startPrev_col_ = prev_col_ - float(half_window);
    const float startPrev_row_ = prev_row_ - float(half_window);
    const int   winWidth_i   = 2*half_window  + 1;
    const int   winHeight_i  = 2*half_window + 1;


    // Calculate sum of gradients
    float  Gxx_f, Gxy_f, Gyy_f, Det_f;
    float          meanLast_f;

    CalculateGradientSumsTex_ZMSSD_32f(dx,dy, previous,
                                       startPrev_col_,
                                       startPrev_row_,
                                       winWidth_i,
                                       winHeight_i,
                                       Gxx_f, Gxy_f, Gyy_f,
                                       meanLast_f );

    // Calculate determinant
    Det_f = Gxx_f * Gyy_f - Gxy_f * Gxy_f;


    // Check if determinant is too small
    if ( Det_f < 0.00000001f )
    {
        features[ index ].status_i = -1;
        return;
    }

    Det_f = 1.0f / Det_f;

    bool success_b=false;
    for ( int i=0; i < num_iterations; ++i )
    {

        // Check that current feature position is in the image
        if ( ( current_col_ < half_window ) ||
             ( current_col_ > (cols-half_window-1) ) ||
             ( current_row_ < half_window ) ||
             ( current_row_ > (rows-half_window-1) ) )
        {
            features[ index ].status_i = -4;
            return;
        }


        // Calculate the start position of the feature window.
        const float startCurr_col_ = current_col_ - float(half_window);
        const float startCurr_row_ = current_row_ - float(half_window);


        // Calculate the b-vector components
        float bx_f, by_f;
        CalculateBVectorTex_ZMSSD_32f(dx,dy,previous,current,
                                      startPrev_col_,
                                      startPrev_row_,
                                      startCurr_col_,
                                      startCurr_row_,
                                      winWidth_i,
                                      winHeight_i,
                                      bx_f,
                                      by_f,
                                      meanLast_f );

        // Calculate displacement
        const float deltaX_f = ( Gyy_f * bx_f - Gxy_f * by_f ) * Det_f;
        const float deltaY_f = ( Gxx_f * by_f - Gxy_f * bx_f ) * Det_f;

        current_col_ += deltaX_f;
        current_row_ += deltaY_f;


        // End iterations if displacement is below threshold
        float len= sqrtf(( deltaX_f* deltaX_f ) +
                ( deltaY_f* deltaY_f ));

        if ( len < min_displacement )
        {
            success_b = true;
            break;
        }
    }

    if ( ! success_b )
    {
        features[ index ].status_i = -3;
        return;
    }



    // Calculate the start position of the feature window.
    const float startCurr_col_ = current_col_ - float(half_window);
    const float startCurr_row_ = current_row_ - float(half_window);

    features[ index ].residuum_f =
            CalculateZSADTex_ZMSSD_32f(dx,dy,previous,current,  startPrev_col_,
                                       startPrev_row_,
                                       startCurr_col_,
                                       startCurr_row_,
                                       winWidth_i,
                                       winHeight_i ) / ( winWidth_i * winHeight_i );



    // Save current feature position
    features[ index ].current_col_ = current_col_/scale;
    features[ index ].current_row_ = current_row_/scale;
}

void  cudaKLT_ZMSSD(const Pyramid< Texture< float, true > >& previous_pyramid,
                    const Pyramid< Texture< float, true > >& current_pyramid,
                    const Pyramid< Texture< float, true > >& dx_pyramid,
                    const Pyramid< Texture< float, true > >& dy_pyramid,
                    Texture<SCUDAKLTFeature_t, true>& features,
                    int                                 half_window,
                    int                                 num_iterations,
                    float                               min_displacement,
                    float                               max_displacement,
                    int stream)
{

    for ( int level = previous_pyramid.levels()-1; level >= 0; --level )
    {

        const auto& prev=previous_pyramid.getImage( level );
        const auto& curr=current_pyramid.getImage( level );
        // Bind textures
        const auto& prev_dx=dx_pyramid.getImage( level );
        const auto& prev_dy=dy_pyramid.getImage( level );

        // Track features
        dim3 dimBlock( 64, 1, 1 );
        dim3 dimGrid(  divUp(features.cols(), dimBlock.x ), 1, 1 );
        CalculateKLT_ZMSSD<<< dimGrid, dimBlock, stream >>>(  prev.texture(),
                                                         prev_dx.texture(),
                                                         prev_dy.texture(),                                                         
                                                         curr.texture(),
                                                         features.data(),
                                                         features.cols(),
                                                         half_window,
                                                         num_iterations,
                                                         min_displacement,
                                                         max_displacement,
                                                         level,
                                                         previous_pyramid.getImage( level ).rows(),
                                                         previous_pyramid.getImage( level ).cols()
                                                         );

    }
}
}

