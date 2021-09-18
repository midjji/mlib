#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>

#include <mlib/datasets/stereo_sample.h>


namespace cvl{


class DaimlerSample: public StereoSample{
public:
    DaimlerSample()=default;
    // dataset contains 1w images, 1f disparity, and 1b label images
    DaimlerSample(std::vector<cv::Mat1w> images,
                  cv::Mat1f disparity_,
                  cv::Mat1b labels,
                  int frameid, double time);


    // StereoSample overrides
    int rows()        const override;
    int cols()        const override;
    int frame_id()    const override;
    int sequence_id() const override;
    double time()     const override;


    cv::Mat1f disparity_image() const override;
    cv::Mat1f grey1f(int i)     const override;
    cv::Mat1w grey1w(int i)     const override;
    cv::Mat1b grey1b(int i)     const override;
    cv::Mat3f rgb3f(int i)      const override;
    cv::Mat3b rgb(int i) const override; // for display, copy yes!
    //Other

    bool is_car(double row, double col) const;
    bool is_car(Vector2d rowcol) const;
    cv::Mat3b show_labels() const;// for visualization, new clone

private:
    float disparity_impl(double row, double col) const override;
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity_; // holds floating point disparities
    cv::Mat1b labels;
    int frameid_;
    double time_;


};
using sDaimlerSample=std::shared_ptr<DaimlerSample>;
std::shared_ptr<StereoSample> convert2StereoSample(std::shared_ptr<DaimlerSample> sd);

}
