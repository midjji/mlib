#pragma once
#include "klt/internal/texture.h"
namespace klt{
void harris_corner_score(const Texture< float, true>&  dx,
                            const Texture< float, true>&  dy,
                             Texture< float, true>&  output, int stream=0);
}


