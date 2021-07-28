#pragma once
#include <opencv2/core.hpp>
#include <mlib/utils/cvl/matrix.h>

namespace cvl{

class StereoSample{
public:    
    StereoSample()=default;
    virtual ~StereoSample();

    explicit StereoSample(int sample_id, double sample_time,
                          cv::Mat3b left,cv::Mat3b right,
                          cv::Mat1f disparity,
                          cv::Mat1f lf,cv::Mat1f rf, int sequence_id);

    int rows() const;
    int cols() const;
    int id() const;
    double time() const;

    // only supports single channel images.
    cv::Mat1f get1f(int i); // real data
    cv::Mat3b rgb(int i); // for display
    cv::Mat3b display_disparity();

    float disparity(int row, int col) const;
    float disparity(double row, double col) const;
    float disparity(Vector2d rowcol) const;
    int sequenceid() const;
private:
    int sample_id;
    double sample_time;
    cv::Mat3b left, right; // for display,
    cv::Mat1f disparity_image;
    cv::Mat1f lf; // grayscale,
    cv::Mat1f rf;
    int sequence_id;
};

class DaimlerSample;
namespace kitti {
class KittiOdometrySample;
}
std::shared_ptr<StereoSample> convert2StereoSample(std::shared_ptr<DaimlerSample> sd);
std::shared_ptr<StereoSample> convert2StereoSample(std::shared_ptr<kitti::KittiOdometrySample> sd);



void show(const std::shared_ptr<StereoSample>& s);


}
