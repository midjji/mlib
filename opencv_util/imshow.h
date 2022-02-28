#pragma once
/* ********************************* FILE ***********************************
 * \file    imshow.h
 *
 * \brief    This file provides simplified/fixed versions of the standard opencv imshow, and waitkey while alleviating compile times
 *
 * \remark
 *
 * \todo
 * - add draw now without wait
 *
 ******************************************************************************/
#include <opencv2/core/mat.hpp>
namespace cvl{
bool imshow(cv::Mat im, std::string name="imshow");
bool imshow(std::string name, cv::Mat im);
void input_window();
uchar waitKey(double time/*0 means inf...*/);

}
