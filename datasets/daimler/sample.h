#pragma once
#include <map>
#include <opencv4/opencv2/core.hpp>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/bounding_box.h>


namespace cvl{


class DaimlerSample{
public:
    DaimlerSample(){}
    // dataset contains 1w images, 1f disparity, and 1b label images
    DaimlerSample(std::vector<cv::Mat1w> images,
            cv::Mat1f dim,
                  cv::Mat1b labels,
            int frameid):images(images), dim(dim),
        labels(labels), frameid_(frameid){}



    float getDim(double row, double col);
    float getDim(Vector2d rowcol);
    Vector3d get_3d_point(double row, double col);
    bool is_car(double row, double col);
    bool is_car(Vector2d rowcol);


    cv::Mat1b disparity_image(); // for visualization, new clone
    cv::Mat3b disparity_image_rgb(); // for visualization, new clone
    cv::Mat3b rgb(uint id); // for visualization, new clone
    cv::Mat1b gray(uint id); // for visualization, new clone
    cv::Mat1f grayf(uint id);
    cv::Mat3b show_labels();// for visualization, new clone
    uint rows();
    uint cols();
    int frameid();


    std::vector<cv::Mat1w> images;
    cv::Mat1f dim; // holds floating point disparities
    cv::Mat1b labels;
    int frameid_;
private:


};
using sDaimlerSample=std::shared_ptr<DaimlerSample>;


}
