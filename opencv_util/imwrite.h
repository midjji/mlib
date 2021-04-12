#pragma once
#include <opencv2/core/mat.hpp>
namespace mlib{
/**
 * @brief write_image_safe
 * @param path
 * @param img
 * @return if successful
 *
 * guarantees that the entire image is written if return true
 * creates the directory if needed,
 */
bool write_image_safe(std::string path, cv::Mat img);
bool write_image_safe_auto_type(std::string path_without_ext, cv::Mat1b img);
bool write_image_safe_auto_type(std::string path_without_ext, cv::Mat3b img);
bool write_image_safe_auto_type(std::string path_without_ext, cv::Mat1w img);

}
