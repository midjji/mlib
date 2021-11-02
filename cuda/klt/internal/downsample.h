#pragma once
#include "klt/internal/texture.h"
bool
downsample_gauss_filter3( const Texture< float, true >& input,
                         Texture< float, true >& output,
                         float scale=1 // the scaling of the values, not the image
        );
bool
downsample_gauss_filter5( const Texture< float, true >& input,
                         Texture< float, true >& output,
                         float  scale=1// the scaling of the values, not the image
        );

void
downsample_max(
        const Texture< float, true >& input,
                         Texture< float, true >& output
        );
