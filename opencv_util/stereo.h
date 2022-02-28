#pragma once
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/matrix.h>
namespace cvl
{
cv::Mat1f stereo(cv::Mat3b l, cv::Mat3b r,int max_disparity);
cv::Mat1f stereo(cv::Mat1b l, cv::Mat1b r,int max_disparity);
cv::Mat1f stereo(cv::Mat1f l, cv::Mat1f r,int max_disparity);

std::vector<double>
sparse_stereo(cv::Mat1f l, cv::Mat1f r, int max_disparity, std::vector<cvl::Vector2d> ys);
std::vector<double>
sparse_stereo(cv::Mat1f l, cv::Mat1f r, int max_disparity, std::vector<cvl::Vector2f> ys);

cv::Mat1b display_disparity(cv::Mat1f disparity);
cv::Mat3b offset_left(cv::Mat3b rgb, int cols);
}
