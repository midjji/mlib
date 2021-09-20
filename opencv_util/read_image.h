#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <future>

namespace mlib{
// internal templateed versions are intentionally not exposed.

// more expressive errors than opencv
cv::Mat1b read_image1b(std::string path) noexcept;
cv::Mat3b read_image3b(std::string path) noexcept;
cv::Mat1w read_image1w(std::string path) noexcept;
cv::Mat1f read_image1f(std::string path) noexcept;

// launched async
std::future<cv::Mat1b> future_read_image1b(std::string path) noexcept;
std::future<cv::Mat3b> future_read_image3b(std::string path) noexcept;
std::future<cv::Mat1w> future_read_image1w(std::string path) noexcept;
std::future<cv::Mat1f> future_read_image1f(std::string path) noexcept;

// paralell read many
std::map<int,cv::Mat1f> read_image1f(std::map<int,std::string> paths) noexcept;


}
