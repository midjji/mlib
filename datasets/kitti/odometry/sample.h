#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/datasets/stereo_sample.h>
namespace cvl {

namespace kitti {
class KittiOdometrySample : public StereoSample
{
public:
    KittiOdometrySample()=default;
    KittiOdometrySample( float128 time,const StereoSequence* ss,
                        int frame_id, std::vector<cv::Mat1f> images,
                        cv::Mat1f disparity_);
    KittiOdometrySample(std::vector<cv::Mat1w> images,
                        cv::Mat1f disparity,     
                        int frameid_, double time_, const StereoSequence* ss);
};


}

}
