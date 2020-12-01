#pragma once
#include <kitti/odometry/eval.h>

namespace cvl{
namespace kitti{


/**
 * @brief tomatlabfile write a matlab script with extended kitti egomotion statistics
 * @param results
 * \todo
 * - rewrite so it uses the toMatlabStruct function
 */
void tomatlabfile(const std::vector<Result>& results);


}// end kitti namespace
}// end namespace cvl
