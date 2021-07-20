#pragma once
#include <opencv2/core/mat.hpp>
namespace cvl {

namespace kitti {
class KittiOdometrySample{
public:

    KittiOdometrySample(std::vector<cv::Mat1w> images,
                        cv::Mat1f disparity,
                        int sequenceid_,
                        int frameid_, double time_);
    int frameid() const;
    int sequenceid()const;
    int rows()const;
    int cols()const;
    double time() const;
    float disparity(double row, double col) const;

    cv::Mat1b disparity_image() const; // for visualization, new clone
    cv::Mat1f disparity_imagef() const; // exact
    cv::Mat3b rgb(uint id);// for visualization, new clone
    cv::Mat1w greyw(uint id);// for visualization, new clone
    cv::Mat1b greyb(uint id);// for visualization, new clone
    cv::Mat1f greyf(uint id);// for visualization, new clone

private:
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity_;
    int sequenceid_;
    int frameid_;
    double time_; // in seconds from start.

};
}
}
