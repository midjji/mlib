#pragma once
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
}
