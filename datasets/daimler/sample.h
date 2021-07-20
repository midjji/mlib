#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/bounding_box.h>


namespace cvl{


class DaimlerSample{
public:
    DaimlerSample()=default;
    // dataset contains 1w images, 1f disparity, and 1b label images
    DaimlerSample(std::vector<cv::Mat1w> images,
                  cv::Mat1f disparity_,
                  cv::Mat1b labels,
                  int frameid, double time);


    // returns -1 for missing or out of image...
    float disparity(double row, double col) const;
    float disparity(Vector2d rowcol) const;
    Vector3d get_3d_point(double row, double col) const;
    bool is_car(double row, double col) const;
    bool is_car(Vector2d rowcol) const;
    double time() const;


    cv::Mat1b disparity_image()const;  // for visualization, new clone
    cv::Mat1f disparity_imagef()const;  // for visualization, new clone
    cv::Mat3b disparity_image_rgb()const; // for visualization, new clone
    cv::Mat3b rgb(uint id)const; // for visualization, new clone
    cv::Mat1b gray(uint id)const; // for visualization, new clone
    cv::Mat1f grayf(uint id)const;
    cv::Mat3b show_labels()const;// for visualization, new clone
    int rows() const;
    int cols() const;
    int frameid() const;



private:
    std::vector<cv::Mat1w> images;
    cv::Mat1f disparity_; // holds floating point disparities
    cv::Mat1b labels;
    int frameid_;
    double time_;


};
using sDaimlerSample=std::shared_ptr<DaimlerSample>;


}
