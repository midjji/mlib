#pragma once
#include "klt/internal/texture.h"


// exact variant
void non_max_supression(
        const Texture< float, true>&  input,
                   Texture< float, true>&  output,
                   float radius);



void non_max_supression_approx(
        const Texture< float, true>&  input,
                   Texture< float, true>&  output,
                   float radius);
