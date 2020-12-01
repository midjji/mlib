#pragma once
#include <opencv2/core/core.hpp>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/colormap.h>
namespace mlib{
void drawbox(cv::Mat3b& im,cvl::Vector2d y0,cvl::Vector2d y1, Color color);
void drawbox(cv::Mat3b& im,int row, int col, int rows, int cols, Color color);
}
