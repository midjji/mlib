#pragma once
#include "klt/internal/texture.h"



void scharr_dx_dy(   const     Texture< float, true >&  input,
                                     Texture< float, true >&  dx,
                                     Texture< float, true >&  dy );
