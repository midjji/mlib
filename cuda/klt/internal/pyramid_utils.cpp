#include "pyramid_utils.h"
#include "downsample.h"
#include "gauss.h"

#include "scharr.h"
#include <iostream>





void pyramid_gauss3x3_gauss5x5(
        const Texture< float, true >&                 input,
        Pyramid< Texture< float, true > >& pyramid,
        int levels )
{
    //mlog()<<levels<<"\n";
    mhere();
    gauss_filter3( input, pyramid.getImage( 0 ) );
    mhere();

    if(levels==1) return;

    downsample_gauss_filter3( pyramid.getImage( 0 ),
                              pyramid.getImage( 1 ) );
    mhere();


    for ( int i = 2; i < levels; ++i )
    {
        downsample_gauss_filter5( pyramid.getImage( i - 1 ),
                                  pyramid.getImage( i ) );
        mhere();

    }

        mhere();
}



void scharr_pyramid(
        const Pyramid< Texture< float, true > >&  pyramid,
        Pyramid< Texture< float, true > >&  dx,
        Pyramid< Texture< float, true > >&  dy)
{


    for ( int i=0;i<pyramid.levels(); ++i )
    {

        scharr_dx_dy( pyramid.getImage( i ),
                      dx.getImage( i ),
                      dy.getImage( i ) );


    }

}

