#include <string>
#include <opencv2/core/mat.hpp>
namespace mlib {

std::string type2str(int cv_mat_type);
template<class T> int Tnum(){ return -1;}
template<> int Tnum<uint8_t>(){ return CV_8U;}
template<> int Tnum<uint16_t>(){ return CV_16U;}
template<> int Tnum<float>(){ return CV_32F;}
template<> int Tnum<cv::Vec3b>(){ return CV_8UC3;}

bool is_mat1b(cv::Mat);
bool is_mat1w(cv::Mat);
bool is_mat1f(cv::Mat);
bool is_mat3b(cv::Mat);
bool is_mat3f(cv::Mat);



}
