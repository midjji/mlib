#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/datasets/image_sample.h>

namespace cvl{
struct StereoSample:public ImageSample
{

    StereoSample(float128 time,
                 const std::shared_ptr<StereoSequence> ss,
                 int frame_id, std::vector<cv::Mat1f> images,
                 cv::Mat1f disparity_);

    virtual ~StereoSample();

    virtual cv::Mat1f disparity_image() const;
    virtual cv::Mat3b display_disparity() const;


    virtual float disparity(double row, double col) const;
    float disparity(Vector2d rowcol) const;
    virtual int type() const override;
    virtual bool complete() const;
    virtual bool has_stereo() const;
    virtual float128 original_time() const{return 0;}


private:
    const cv::Mat1f disparity_;
};



void show(const std::shared_ptr<StereoSample>& s);


}
