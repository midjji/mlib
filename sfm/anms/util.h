#pragma once
#include <mlib/sfm/anms/base.h>
#include <mlib/opencv_util/cv.h>

namespace cvl{
namespace anms{

cv::Mat3b drawData(std::vector<anms::Data>& datas);
void show(std::vector<Data>& datas,std::string name);








} // end namespace anms
}// end namespace cvl

