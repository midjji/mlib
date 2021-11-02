#pragma once
#include "klt/internal/texture.h"

bool estimate_mean(
        Texture< float, true >& input,
                  Texture< float, true >&       output,
                  int half_width,
                  int half_height);


