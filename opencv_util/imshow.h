#pragma once
#include <opencv2/core.hpp>
namespace cvl{
bool imshow(cv::Mat im, std::string name="imshow");
bool imshow(std::string name, cv::Mat im);
}
