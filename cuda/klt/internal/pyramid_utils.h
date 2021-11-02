#pragma once
#include "klt/internal/texture.h"
#include "klt/internal/pyramid.h"



void
pyramid_gauss3x3_gauss5x5(
        const Texture< float, true >&                image,
        Pyramid< Texture< float, true > >& pyramid,
        int levels );

void scharr_pyramid(
        const Pyramid< Texture< float, true > >&  pyramid_image,
        Pyramid< Texture< float, true > >&  pyramid_dx,
        Pyramid< Texture< float, true > >&  pyramid_dy );

