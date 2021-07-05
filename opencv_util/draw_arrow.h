#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/colormap.h>


namespace mlib{


void drawArrow(cv::Mat3b im,
               cvl::Vector2d from, // row,col
               cvl::Vector2d to, // row,col
               Color col=Color::cyan(),
               int thick=1,
               int length=5);
void draw_legend(cv::Mat3b rgb, std::vector<std::tuple<std::string,Color>> labels);

cv::Mat3b draw_arrows(const std::vector<std::pair<cvl::Vector2d,cvl::Vector2d>>& deltas,
               cv::Mat3b rgb);
}


