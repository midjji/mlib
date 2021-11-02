#pragma once
#include "klt/internal/texture.h"


// exact variant
void non_max_supression(
        const Texture< float, true>&  input,
                   Texture< float, true>&  output,
                   float radius);



