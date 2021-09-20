#pragma once
#include <opencv2/core/mat.hpp>
namespace cvl {

cv::Mat3b image2rgb3b(const cv::Mat1b& img,
               float scale=1,
               float offset=0);
cv::Mat3b image2rgb3b(const cv::Mat1f& img,
               float scale=1,
               float offset=0);

cv::Mat3f image2rgb3f(const cv::Mat1w& img,
                      float scale=1,
                      float offset=0);
cv::Mat1f image2grey1f(const cv::Mat1w& img,
                      float scale=1,
                      float offset=0);
cv::Mat1w image2grey1w(const cv::Mat1b& img,
                      float scale=16,
                      float offset=0);
cv::Mat1b image2grey1b(const cv::Mat1w& img,
                      float scale=1,
                      float offset=0);



std::vector<cv::Mat1f> images2grey1f(std::vector<cv::Mat1w> imgs,  float scale=1,
                                    float offset=0);

}
