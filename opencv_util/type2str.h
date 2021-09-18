#pragma once
#include <string>
#include <opencv2/core/mat.hpp>
namespace mlib {

std::string type2str(int cv_mat_type);
template<class T> int Tnum(){ return -1;}
template<> int Tnum<uint8_t>();
template<> int Tnum<uint16_t>();
template<> int Tnum<float>();
template<> int Tnum<cv::Vec3b>();

bool is_mat1b(cv::Mat);
bool is_mat1w(cv::Mat);
bool is_mat1f(cv::Mat);
bool is_mat3b(cv::Mat);
bool is_mat3f(cv::Mat);



}
