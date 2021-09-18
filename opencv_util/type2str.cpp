#include "mlib/opencv_util/type2str.h"
#include <opencv2/core/mat.hpp>
namespace mlib {

std::string type2str(int cv_m_type)
{
    switch(cv_m_type){
    case CV_8U:        return "8U";
    case CV_8S:        return "8S";
    case CV_16U:        return "16U";
    case CV_16S:        return "16S";
    case CV_32S:        return "32S";
    case CV_32F:        return "32F";
    case CV_64F:        return "64F";
    case CV_8UC3:        return "8UC3";
    default: break;
    }
 return "unknown";
}
bool is_mat1b(cv::Mat im){   return im.type()==CV_8U;}
bool is_mat1w(cv::Mat im){   return im.type()==CV_16U;}
bool is_mat1f(cv::Mat im){   return im.type()==CV_32F;}
bool is_mat3b(cv::Mat im){   return im.type()==CV_8UC3;}
bool is_mat3f(cv::Mat im){   return im.type()==CV_32FC3;}

}
