#pragma once
#include "klt/internal/texture.h"
#include "klt/internal/pyramid.h"
#include "klt/feature.h"

namespace klt{

void
cudaKLT( const Pyramid< Texture< float, true > >& previous_pyramid,
         const Pyramid< Texture< float, true > >& current_pyramid,
         const Pyramid< Texture< float, true > >& dx_pyramid,
         const Pyramid< Texture< float, true > >& dy_pyramid,
         Texture<SCUDAKLTFeature_t, true>& features,
         int half_window,
         int num_iterations,
         float min_displacements,
         float max_displacement,
         int stream=0);


void cudaKLT_ZMSSD( const Pyramid< Texture< float, true > >& previous_pyramid,
                    const Pyramid< Texture< float, true > >& current_pyramid,
                    const Pyramid< Texture< float, true > >& dx_pyramid,
                    const Pyramid< Texture< float, true > >& dy_pyramid,
                    Texture<SCUDAKLTFeature_t, true>& features,
                    int half_window,
                    int num_iterations,
                    float min_displacements, float max_displacement,
                    int stream=0 );



}
