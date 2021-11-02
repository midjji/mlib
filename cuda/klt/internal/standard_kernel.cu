#include <cstdio>

#include <mlib/utils/mlog/log.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/cvl/matrix.h>

#include "klt/internal/common.h"
#include "klt/internal/klt_kernels.h"



namespace klt{

namespace kernel {



__device__ inline float
residual( const cudaTextureObject_t pi,
          const cudaTextureObject_t ci,
          const cvl::Vector2f p_x,
          const cvl::Vector2f c_x,
          int half_window)
{
    // its the mean of the squared intensity error over the patches.
    float residual_ = 0.0f;
    for ( int r=-half_window; r < half_window+1; ++r ) {
        for ( int c=-half_window; c < half_window+1; ++c ) {


            float err = tex2D<float>( pi, p_x[1] +c, p_x[0] +r ) -
                    tex2D<float>( ci, c_x[1] +c, c_x[0] +r );
            residual_      += err*err;
        }
    }
    return sqrtf(residual_);
}


__device__ inline cvl::Vector3f
compute_z(
        const cudaTextureObject_t dx,
        const cudaTextureObject_t dy,
        const cvl::Vector2f position,
        const int       half_window)
{
    cvl::Vector3f zs(0.0f,0.0f,0.0f); // xx, xy, yy


    for (int y=-half_window; y < half_window+1; ++y )
    {
        for (int x=-half_window; x < half_window; ++x )
        {
            const float xv = position[1] + float(x);
            const float yv = position[0] + float(y);

            const float dx_ = tex2D<float>(dx, xv, yv);
            const float dy_ = tex2D<float>(dy, xv, yv);

            zs[0] += ( dx_* dx_ );
            zs[1] += ( dx_* dy_ );
            zs[2] += ( dy_* dy_ );

        }
    }
    return zs;
}


__device__ inline cvl::Vector2f
residualxjacobian(
        const cudaTextureObject_t dx,
        const cudaTextureObject_t dy,
        const cudaTextureObject_t pi,
        const cudaTextureObject_t ci,
        const cvl::Vector2f start_last,
        const cvl::Vector2f start_curr,
        const int   half_window)
{
    cvl::Vector2f b(0.0f,0.0f);
    for ( int y=-half_window; y < half_window+1; ++y )
    {
        for ( int x=-half_window; x < half_window+1; ++x )
        {
            float prev_row = start_last[0] + float(y);
            float curr_row = start_curr[0] + float(y);
            float prev_col = start_last[1] + float(x);
            float curr_col = start_curr[1] + float(x);
            float dx_= tex2D<float>( dx, prev_col, prev_row );
            float dy_ = tex2D<float>( dy, prev_col, prev_row );
            float delta = tex2D<float>( pi, prev_col, prev_row ) -
                    tex2D<float>( ci, curr_col, curr_row );
            b[0]+=( delta* dy_ );
            b[1]+=( delta* dx_ );
        }
    }
    return b;
}





__global__
/**
 * @brief CalculateKLT
 * @param dx for previous
 * @param dy for previous
 * @param previous
 * @param current
 * @param features
 * @param num_features
 * @param half_window, or radius
 * @param num_iterations
 * @param max_displacement
 * @param min_displacement
 * @param level
 * @param rows
 * @param cols
 *
 *
 * This one is designed to mirror the clemens one, with minor bugfixes,
 * but has been reimplemented from scratch.
 *
 */
void CalculateKLT(
        const cudaTextureObject_t previous,
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
        int                 cols)
{



    // Index according to current thread
    int index = ( blockIdx.x* blockDim.x ) + threadIdx.x;
    if ( index >= num_features ) return;
    SCUDAKLTFeature_t  feature = features[ index ];
    // Abort if feature is not tracked (any more)
    if ( feature.status_i != 0 )  return;

    cvl::Vector2f p_x(feature.prev_row_, feature.prev_col_);
    cvl::Vector2f c_x(feature.current_row_,feature.current_col_);

    // Get feature data in current pyramid level
    const float scale = 1.0f / float( 1 << level );
    p_x*=scale;
    c_x*=scale;
    if(!(p_x.in(0,0,rows,cols)&&
         c_x.in(0,0,rows,cols))){
        features[ index ].status_i = -1;
        return;
    }

    // gradient outer product matrix

    cvl::Vector3f Z=compute_z(dx, dy,
                              p_x,
                              half_window);

    float determinant = Z[0] * Z[2] - Z[1] * Z[1];


    // Check if determinant is too small
    if ( determinant < 0.00000001f )
    {
        features[ index ].status_i = -1;
        return;
    }

    for ( int i=0; i < num_iterations; ++i )
    {

        if(!c_x.in(0,0,rows,cols)){
            features[ index ].status_i = -2;
            return;
        }


        cvl::Vector2f b=residualxjacobian(dx, dy, previous, current,
                                          p_x, c_x,
                                          half_window);
        // delta= Z^-{-1}*b
        cvl::Vector2f delta(
                    ( Z[0] * b[0] - Z[1] * b[1] ) /determinant,
                ( Z[2] * b[1] - Z[1] * b[0] ) /determinant);
        // do not take steps longer than 1 pixel,
        // the derivatives are likely not valid for that...
        float len=sqrtf(delta.squaredNorm());
        if(len>max_displacement){
            delta/=len;
            delta*=max_displacement;
        }

        c_x+=delta;//*0.9f;// dont over shoot?
        // End iterations if displacement is below threshold
        if ( len < min_displacement ) {

            features[ index ].residuum_f = residual(
                        previous, current,
                        p_x,
                        c_x,
                        half_window);



            // Save current feature position
            features[ index ].current_row_ = c_x[0]/scale;
            features[ index ].current_col_ = c_x[1]/scale;
            return;
        }
    }
// did not converge in time
    features[ index ].status_i = -1;



}

}

mlib::NamedTimerPack ntp2;
void cuda_klt( const Texture< float, true >& prev,
               const Texture< float, true > & prev_dx,
               const Texture< float, true > & prev_dy,
               const Texture< float, true > & curr,
               Texture<SCUDAKLTFeature_t, true>& features,
               int                 half_window,
               int                 num_iterations,
               float               min_displacement,
               float               max_displacement,
               int                 level,
               int stream )
{



    // Track features
    dim3 dimBlock( 64, 1, 1 );
    dim3 dimGrid(  divUp( features.cols(), dimBlock.x ), 1, 1 );

    kernel::CalculateKLT<<< dimGrid, dimBlock, stream >>>(
                                                            prev.texture(),
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
                                                            prev.rows(),
                                                            prev.cols());





}


void
cudaKLT( const Pyramid< Texture< float, true > >& previous_pyramid,
         const Pyramid< Texture< float, true > >& current_pyramid,
         const Pyramid< Texture< float, true > >& dx_pyramid,
         const Pyramid< Texture< float, true > >& dy_pyramid,
         Texture<SCUDAKLTFeature_t, true>& features,
         int                                 half_window,
         int                                 num_iterations,
         float                               min_displacement,
         float                               max_displacement,
         int stream )
{
    mlog()<<half_window<<" "<<num_iterations<<" "<<min_displacement<<" "<<max_displacement<<"\n";

    // 1 ms at 2000, 20 rad, at daimler sequence,
    // 4ms at 20000 10 rad, ...
    mhere();
    ntp2["klt"].tic();
    for ( int level = previous_pyramid.levels() -1; level >= 0; --level )
    {
        const auto& prev=previous_pyramid.getImage( level );
        const auto& curr=current_pyramid.getImage( level );
        // Bind textures
        const auto& dx=dx_pyramid.getImage( level );
        const auto& dy=dy_pyramid.getImage( level );

        cuda_klt(prev, dx, dy, curr,
                 features,
                 half_window,
                 num_iterations,
                 min_displacement,
                 max_displacement,
                 level,
                 stream);
    }
    mhere();
    ntp2["klt"].toc();
    std::cout<<ntp2<<std::endl;
}
}
