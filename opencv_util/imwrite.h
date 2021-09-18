#pragma once
#include <map>
#include <future>

#include <opencv2/core/mat.hpp>

namespace mlib{
/**
 * @brief write_image_safe
 * @param path
 * @param img
 * @return if successful
 *
 * guarantees that the entire image is written if return true
 * writes to unique intermediate file first, then moves
 * creates the directory if needed,
 *
 * Will warn on extension not matching image type.
 *
 *
 */
bool write_image_safe(std::string path, cv::Mat img) noexcept;
std::future<bool> future_write_image(std::string path, cv::Mat img) noexcept;
std::map<std::string, bool> future_write_image(std::map<std::string, cv::Mat> images) noexcept;
bool all_good(const std::map<std::string, bool>& fwis) noexcept;

}
