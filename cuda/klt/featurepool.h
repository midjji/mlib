#pragma once

#include <cstdint>
#include <sstream>
#include <vector>

#include "mlib/cuda/klt/feature.h"
#include <mlib/utils/cvl/matrix.h>
#include <opencv2/core/mat.hpp>
namespace klt {



std::ostream& operator<<(std::ostream &os, SFeature_t);

class FeaturePool
{
    //------------------------------------------------------------------
public:
    std::string str() const;
    std::vector<cvl::Vector2f> valid_kps() const;
    void assign_new(const
            std::vector<cvl::Vector2f>& kps,
            bool grow_if_required=false);

    FeaturePool()=default;

    // Constructor with given initial size.
    FeaturePool( int initial_size, int reserve );
    /// Get the current size of the feature pool.
    SFeature_t* begin();
    const SFeature_t* begin() const;
    const SFeature_t* end() const;

    template<class Vector2d> void set_points_to_track(const std::vector<Vector2d>& ys)
    {
        for(uint i=0;i<pool.size();++i)
        {
            auto& f=pool[i];
            f.clear();

            if(i<ys.size()) {
                f.set(ys[i][0],ys[i][1],-1,-1);
            }
        }
    }
    template<class Vector2d> void set_points_to_track(const std::vector<Vector2d>& ys,
                                                      const std::vector<Vector2d>& yps)
    {
        for(uint i=0;i<pool.size();++i)
        {
             auto& f=pool[i];
             f.clear();

            // then if we have a feature to track,
            if(i>=ys.size()) continue;
            const auto& y=ys[i];

            if(i<yps.size()) // and a prediction
                f.set(y,yps[i]);
            else
                f.set(y,y);
        }
    }


    int  size() const;
    void reserve(int size);
    void resize( int new_size );
    void reset();
    void clear_lost();
    SFeature_t& operator[](int index);
    const SFeature_t& operator[](int index) const;
    void print() const;
    std::vector<SFeature_t> pool;
};

cv::Mat3b draw_feature_pool(FeaturePool& pool,
                            cv::Mat3b rgb);

cv::Mat3b draw_feature_pool_prediction(const FeaturePool& pool, cv::Mat3b rgb);

}
