#pragma once
#include "klt/internal/texture.h"


bool gauss_filter3(     const  Texture< float, true >& input,
                     Texture< float, true >& output );
bool gauss_filter5( const      Texture< float, true >& input,
                     Texture< float, true >& output );
