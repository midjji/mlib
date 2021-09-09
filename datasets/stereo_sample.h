#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/matrix.h>

namespace cvl{

struct StereoSample{

    StereoSample()=default;
    virtual ~StereoSample();

    virtual int rows() const=0;
    virtual int cols() const=0;
    virtual int frame_id() const=0;    
    virtual int sequence_id() const=0;
    virtual double time() const=0;

    // only supports single channel images.
    virtual cv::Mat1b grey1b(int i) const=0; // for display, copy yes!
    virtual cv::Mat1f grey1f(int i) const =0; // copy?
    virtual cv::Mat3f rgb3f(int i) const=0;// copy?
    virtual cv::Mat3b rgb(int i) const=0; // for display, copy yes!


    virtual cv::Mat1f disparity_image() const=0;
    virtual cv::Mat3b display_disparity() const;


    virtual float disparity(double row, double col) const;
    float disparity(Vector2d rowcol) const;

protected:
    virtual float disparity_impl(double row, double col) const=0;
};



void show(const std::shared_ptr<StereoSample>& s);


}
