#pragma once
#include <vector>
#include "mlib/opencv_util/cv.h"
namespace cvl{
std::vector<std::vector<cv::DMatch>>
matchBrief(cv::Mat query, cv::Mat train, int maxdist=96);
}
