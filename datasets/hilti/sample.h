#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>
using float128=long double;
static_assert (sizeof(float128)==16,"must be big enough");
#include <mlib/datasets/stereo_sample.h>


namespace cvl{
namespace hilti {

struct Sample
{
    float128 time() const{return time_;}

    Sample(float128 time_):time_(time_){}
    virtual ~Sample(){}
    virtual void show() const=0;
private:
        float128 time_;
};


/**
 * @brief The ImageSample class
 */
struct ImageSample:public Sample
{
    // dataset contains 1w images, 1f disparity, and 1b label images
    ImageSample(float128 time, std::map<int,cv::Mat1b> images):Sample(time),images(images){}
    cv::Mat3b rgb(int i) const;
    cv::Mat1f grey1f(int i) const;
    bool complete() const;
    void show() const override;
private:

    // the raw images
    // not all are guaranteed to exist
    std::map<int,cv::Mat1b> images;
};



}
}
