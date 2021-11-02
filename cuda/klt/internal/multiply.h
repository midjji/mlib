#pragma once
#include "klt/internal/texture.h"


bool set2( Texture< float, true >&  image, float val);
bool set2( Texture< float, false >&  image, float val);
void multiply( const Texture< float, true>&  input,
                     const   Texture< float, true>&  factor,
                       Texture< float, true>&  output );



