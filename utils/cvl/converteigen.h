#pragma once
// eigen is column major!
//#pragma GCC diagnostic push
#include <Eigen/Core>// no need
#include <Eigen/Geometry>
//#pragma GCC diagnostic pop


#include <mlib/utils/cvl/pose.h>

namespace Eigen {
// Can I forward declare them? well they are strange...
}

namespace cvl{




template<class T, int Rows, int Cols>
cvl::Matrix<T,Rows,Cols> convert2Cvl(const Eigen::Matrix<T, Rows, Cols>& M){
    cvl::Matrix<T,Rows,Cols> ret(&M(0,0));
    // swaps the eigen col major to row major
    return ret.transpose();
}

template<class T, int Rows, int Cols>
Eigen::Matrix<T, Cols, Rows> convert2Eigen(const cvl::Matrix<T,Rows,Cols>& M){
    // convert to col major
    cvl::Matrix<T,Cols, Rows> tmp=M.transpose();
    Eigen::Matrix<T, Cols, Rows> ret;
    for(int c=0;c<Cols;++c)
        for(int r=0;r<Rows;++r)
            ret(r,c)=tmp(r,c);
    return ret;
}

// Transform<double,3,Isometry> is Isometry3d
template<class T>
cvl::Pose<T> convert2cvl(const Eigen::Transform<T,3,Eigen::Isometry>& iso){
    cvl::Matrix4<T> M;
    for(uint i=0;i<4;++i)
        for(uint j=0;j<4;++j)
            M(i,j)=iso(i,j);
    return cvl::Pose<T>(M);
}

template<class T>
Eigen::Transform<T,3,Eigen::Isometry> convert2isometry(const cvl::Pose<T>& pose){
    Eigen::Transform<T,3,Eigen::Isometry> iso;
    cvl::Matrix4<T> M=pose.get4x4();
    for(uint i=0;i<4;++i)
        for(uint j=0;j<4;++j)
            iso(i,j)=M(i,j);
    return iso;
}








}// end namespace cvl
