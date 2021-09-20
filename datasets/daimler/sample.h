#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>

#include <mlib/datasets/stereo_sample.h>


namespace cvl{


class DaimlerSample: public StereoSample{
public:

    // dataset contains 1w images, 1f disparity, and 1b label images
    DaimlerSample(float128 time_,const StereoSequence* ss,
                  int frame_id, std::vector<cv::Mat1f> images, cv::Mat1f disparity,
                  cv::Mat1b labels);
    DaimlerSample(std::vector<cv::Mat1w> images,
                  cv::Mat1f disparity_,
                  cv::Mat1b labels,
                  int frameid, double time,const StereoSequence* ss);

    bool is_car(double row, double col) const;
    bool is_car(Vector2d rowcol) const;
    cv::Mat3b show_labels() const;// for visualization, new clone

private:   
    cv::Mat1b labels;
    int frameid_;
    double time_;


};
using sDaimlerSample=std::shared_ptr<DaimlerSample>;


}
