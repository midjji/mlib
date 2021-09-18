#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/datasets/stereo_sample.h>
namespace cvl {

namespace kitti {
class KittiOdometrySample : public StereoSample
{
public:
    KittiOdometrySample()=default;
    KittiOdometrySample(std::vector<cv::Mat1w> images,
                        cv::Mat1f disparity,
                        int sequenceid_,
                        int frameid_, double time_);
    // StereoSample overrides
    // StereoSample overrides
    int rows()        const override;
    int cols()        const override;
    int frame_id()    const override;
    int sequence_id() const override;
    double time()     const override;


    cv::Mat1f disparity_image() const override;
    cv::Mat1f grey1f(int i)     const override;
    cv::Mat1b grey1b(int i)     const override;
    cv::Mat1w grey1w(int i)     const override;
    cv::Mat3f rgb3f(int i)      const override;
    cv::Mat3b rgb(int i)        const override; // for display, copy yes!

private:
    float disparity_impl(double row, double col) const override;
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity_;
    int sequenceid_;
    int frameid_;
    double time_; // in seconds from start.

};


}

}
