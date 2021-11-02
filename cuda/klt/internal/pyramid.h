#pragma once
#include <map>
#include <memory>
#include <sstream>
#include "klt/internal/texture.h"


template< class ImageType >
class Pyramid
{
    //------------------------------------------------------------------
public:
    std::map< int, std::shared_ptr<ImageType>> images;
    Pyramid()=default;


    int levels() const { return images.size(); }


    /// Get a reference to the Texture of the specified level.
    ImageType& getImage( int level = 0 )
    {

        auto& it=images[ level ];
        if(it==nullptr) it = std::make_shared<ImageType>();
        return *images[level];
    }

    /// Get a reference to the Texture of the specified level.
    const ImageType& getImage( int level = 0 ) const {
        auto it=images.find(level);
        if(it!=images.end()) return *it->second;
        mlog()<<"dont\n";
        exit(1);
    }
    std::string str() const{
        std::stringstream ss;
        for(const auto& [a,b]:images ){
            ss<<a<<", "<<b->str()<<"\n";
        }
        return ss.str();
    }
    // cant copy construct these
    Pyramid(const Pyramid&) =delete;

};

