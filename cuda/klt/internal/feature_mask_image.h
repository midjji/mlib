#pragma once

#include "klt/internal/texture.h"
#include <mlib/utils/cvl/matrix.h>


void feature_mask(
        Texture< float,true>&  mask, // change to char later...
        const Texture< float, true>& features,
                  float radius);
