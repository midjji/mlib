#pragma once


#include <mlib/utils/informative_asserts.h>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/pose.h>
typedef unsigned int uint;

namespace mlib{

bool almost(float a, float b){return std::abs(a-b)<1e-6;}
bool almost(double a, double b){return std::abs(a-b)<1e-12;}
template<unsigned int Rows, unsigned int Cols> bool
almost(cvl::Matrix<float,Rows,Cols> a, cvl::Matrix<float,Rows,Cols> b){
    return almost((a-b).abs().sum(),0.0f);
}
template<unsigned int Rows, unsigned int Cols> bool
almost(cvl::Matrix<double,Rows,Cols> a, cvl::Matrix<double,Rows,Cols> b){
    return almost((a-b).abs().sum(),0.0);
}




}// end namespace mlib

