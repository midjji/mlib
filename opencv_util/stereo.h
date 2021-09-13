#pragma once
#include <opencv2/core/mat.hpp>

namespace cvl{
cv::Mat1f stereo(cv::Mat3b l, cv::Mat3b r,int max_disparity);
cv::Mat1f stereo(cv::Mat1b l, cv::Mat1b r,int max_disparity);
cv::Mat1f stereo(cv::Mat1f l, cv::Mat1f r,int max_disparity);
cv::Mat1b display_disparity(cv::Mat1f disparity);
cv::Mat3b offset_left(cv::Mat3b rgb, int cols);
}
