#pragma once
#include <cstdint>
#include "klt/internal/texture.h"


void
convert( const         Texture< uint16_t, true >& fr_inputImage,
                     Texture< float, true >&    fr_outputImage );
void
convert(  const        Texture< uint8_t, true >&  fr_inputImage,
                     Texture< uint16_t, true >& fr_outputImage );
void
convert(  const        Texture< uint8_t, true >& fr_inputImage,
                     Texture< float, true >&   fr_outputImage );
