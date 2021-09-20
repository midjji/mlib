#pragma once
#include <mlib/datasets/sample.h>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/matrix.h>


namespace cvl{

struct ImageSample:public Sample{


    ImageSample(float128 time,const StereoSequence* ss, int frame_id, std::vector<cv::Mat1f> im);
    virtual ~ImageSample();

    virtual int rows() const;
    virtual int cols() const;
    virtual int frame_id() const;

    // only supports single channel image sets for now...
    cv::Mat1b grey1b(int i=0) const; // for display, copy yes!
    virtual cv::Mat1f grey1f(int i=0) const ; // copy?
    cv::Mat3b rgb(int i=0) const; // for display, copy yes!
    virtual int type() const override;
private:
    int frame_id_;
    std::vector<cv::Mat1f> images;
    // or cv::Mat3f when we finally get such, its always one of the two...

    // weak pointer to the sequence, or calibration?
};




}
