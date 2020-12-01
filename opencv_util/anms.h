#pragma once
typedef unsigned int uint;
#include <vector>
#include "mlib/opencv_util/cv.h"

namespace mlib{

// many forms should be avail, for now just the convenient opencv one
// recall that returning vectors is no problem for c++11 optimized code but for c++ its slower!

/**
 * @brief anms - Adaptive Nonmaximal Suppression( Ideal, slow)
 * @param kps
 * @param radius
 * @return filtered kps
 * This is the ideal && slow version, will sort the kps though
 */
std::vector<cv::KeyPoint> anms(std::vector<cv::KeyPoint>& kps,
                               double radius);

/**
 * @brief anms - Adaptive Nonmaximal Suppression( Ideal, slow)
 * @param kps
 * @param radius
 * @param goal
 * @return filtered kps
 * This is the ideal && slow version, will sort the kps though
 * increases the radius by 5 untill the number is less than goal*1.2
 * slow...
 */
std::vector<cv::KeyPoint> anms(std::vector<cv::KeyPoint>& kps, double radius, int goal);
/**
 * @brief anms - Adaptive Nonmaximal Suppression( Ideal, slow)
 * @param lockedkps  - these are included first, not filtered then the rest
 * @param kps
 * @param radius
 * @return
 */
std::vector<cv::KeyPoint> anms(std::vector<cvl::Vector2d>& lockedkps,std::vector<cv::KeyPoint>& kps,double radius);
/**
 * @brief aanms - Approximate Adaptive Nonmaximal Suppression(a little less Ideal, faster but not perfect yet) rittums1.0 has a faster one
 * @param kps
 * @param radius
 * @return
 */
std::vector<cv::KeyPoint> aanms(std::vector<cv::KeyPoint>& kps, double radius);

/**
 * @brief FastAnms - Adaptive Non-maximal Suppression - fast version
 *
 * @param keypoints    keypoints to filter
 * @param width  width of the image the keypoints were created from
 * @param height height of the image the keypoints were created from
 * @param radius filter radius (pixels)
 * @return filtered keypoints
 *
 * @note This is the fast version, approximately 6 - 35 times faster than
 * the slow && ideal implementation.
 * @note FIXME: The filter area around the keypoint might be asymmetric -
 * typically off by one pixel.
 */
std::vector<cv::KeyPoint> FastAnms(std::vector<cv::KeyPoint>& keypoints, int height, int width, float radius);

// variants, goal, l1, use known image size( needed?)






}// end namespace mlib


