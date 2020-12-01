#pragma once
#include <exception>
#include <stdexcept>

#include <opencv2/core.hpp>

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/cvl/matrix_adapter.h>

namespace cvl{
template<class T>
/**
  * @brief convert2Mat
  * @param M
  * @return
  *
  * cv::Mat does not take ownership!
  * opencv lacks take ownership of pointer!
  * \todo why does opencv mat not work with My Matrixes?
  */
cv::Mat_<T> convert2Mat(MatrixAdapter<T> M){
    return cv::Mat_<T>(M.rows,M.cols,M.getData(),(long unsigned int)M.stride);
}
/*
template<class T, class U>
void convert2MatTU(const MatrixAdapter<T>& M, cv::Mat&_<U> ret){
    // will only work if the converter is defined! take a lambda?
    cv::Mat_<U> out=cv::Mat_<U>(M.rows,M.cols);
    for(int r=0;r<M.rows;++r)
        for(int c=0;c<M.cols;++c){
            U u;
            out(r,c)=to<U,T>(M(r,c));
        }
    ret=out;
}

*/

template<class T> inline int getCVType(){ return 4000000;}

template<class T, unsigned int Rows, unsigned int Cols>
cvl::Matrix<T,Rows,Cols> convert2Cvl(const cv::Mat_<T>& m){

    assert(m.rows==Rows);
    assert(m.cols==Cols);


    // memcopy has alignment issues!
    cvl::Matrix<T,Rows,Cols> ret;
    for(uint r=0;r<Rows;++r)
        for(uint c=0;c<Cols;++c)
            ret(r,c)=m(r,c);
    return ret;
}

template<class T, unsigned int Rows, unsigned int Cols>
cv::Mat_<T> convert2Mat(const cvl::Matrix<T,Rows,Cols>& m){

    assert(m.rows()==Rows);
    assert(m.cols()==Cols);
    cv::Mat_<T> ret(Rows,Cols);
    for(uint r=0;r<Rows;++r)
        for(uint c=0;c<Cols;++c)
            ret(r,c)=m(r,c);
    return ret;
}

template<class T>
Vector2<T> convert2Cvl(const cv::Point_<T>& m){
    return Vector2<T>(m.x,m.y);
}

template<class T>
cv::Point_<T> convert2Point(const Vector2<T>& m){
    return cv::Point_<T>(m(0),m(1));
}
template<class T>
cv::Vec<T,2> convert2Vec(const Vector2<T>& m){
    return cv::Vec<T,2>(m(0),m(1));
}
template<class T>
cv::Vec<T,3> convert2Vec(const Vector3<T>& m){
    return cv::Vec<T,3>(m(0),m(1),m(2));
}




/// will copy data
template<class TO, class FROM> cv::Mat_<TO> toMat_(const cv::Mat& m){

    assert(m.type()==getCVType<FROM>());
    if(m.type()!=getCVType<FROM>())
        throw new std::invalid_argument("matrix type does not match, check if specialization exists");
    cv::Mat_<TO> ret(m.rows,m.cols);
    for(int r=0;r<m.rows;++r)
        for(int c=0;c<m.cols;++c)
            ret(r,c)=TO(m.at<FROM>(r,c));
    return ret;
}





cvl::Matrix<double,3,3> inline convert2Cvl3x3D(const cv::Mat& m){
    return convert2Cvl<double,3,3>(toMat_<double,double>(m));
}





template<> inline int getCVType<unsigned char>(){return CV_8U;}
template<> inline int getCVType<char>(){return CV_8S;}
template<> inline int getCVType<unsigned short>(){return CV_16U;}
template<> inline int getCVType<short>(){return CV_16S;}
template<> inline int getCVType<int>(){return CV_32S;}
template<> inline int getCVType<float>(){return CV_32F;}
template<> inline int getCVType<double>(){return CV_64F;}
template<> inline int getCVType<cv::Vec3b>(){return CV_8UC3;}

template<class T> std::string getCVTypeName(T typenr){
    // should be an int...
// possible with macros, but really ugly...
    if(typenr==CV_8U) return "CV_8U";
    if(typenr==CV_8S) return "CV_8S";
    if(typenr==CV_16U) return "CV_16U";
    if(typenr==CV_16S) return "CV_16S";
    if(typenr==CV_32S) return "CV_32S";
    if(typenr==CV_32F) return "CV_32F";
    if(typenr==CV_64F) return "CV_64F";
    if(typenr==CV_8UC3) return "CV_8UC3";
    if(typenr==CV_8UC4) return "CV_8UC4";
    if(typenr==CV_32FC3) return "CV_32FC3";
    if(typenr==CV_32FC4) return "CV_32FC4";
    assert(false && "type not found");
    return "unknown typenr";
}
template<typename T1, typename T2> inline
cvl::Vector2<T1> to(const cv::Point_<T2>& m){
    return cvl::Vector2<T1>(m.x,m.y);
}
template<typename T1, typename T2> inline
cvl::Vector3<T1> to(const cv::Point3_<T2>& m){
    return cvl::Vector3<T1>(m.x,m.y,m.z);
}
template<typename T1, typename T2> inline
cv::Point_<T1> to(const cvl::Vector2<T2>& m){
    return cv::Point_<T1>(m.x,m.y);
}
template<typename T1, typename T2> inline
cv::Point3_<T1> to(const cvl::Vector3<T2>& m){
    return cv::Point3_<T1>(m.x,m.y,m.z);
}

template<typename T>
 cv::Mat_<T> to3x3Mat(cvl::Matrix3<T> m){
    cv::Mat_<T> M=(cv::Mat_<T>(3,3)
               <<    m(0,0), m(0,1), m(0,2),
               m(1,0), m(1,1), m(1,2),
               m(2,0), m(2,1), m(2,2));
    return M;
}
template<typename T>
 cvl::Matrix3<T> from3x3Mat(const cv::Mat& m){
    cvl::Matrix3<T> M;
    assert(m.rows==3);
    assert(m.cols==3);
    M(0,0)=m.at<T>(0,0);    M(0,1)=m.at<T>(0,1);    M(0,2)=m.at<T>(0,2);
    M(1,0)=m.at<T>(1,0);    M(1,1)=m.at<T>(1,1);    M(1,2)=m.at<T>(1,2);
    M(2,0)=m.at<T>(2,0);    M(2,1)=m.at<T>(2,1);    M(2,2)=m.at<T>(2,2);

    return M;
}
template<typename T>
 cvl::Matrix3<T> from1x9Mat(cv::Mat_<T> m){
       return cvl::Matrix3<T>(m(0),m(1),m(2),
                              m(3),m(4),m(5),
                              m(6),m(7),m(8));
    }
template<typename T>
 cvl::Vector3<T> from1x3Mat(cv::Mat_<T> m){
     return cvl::Vector3<T>(m(0),m(1),m(2));
}
 template<class T> cv::Mat1d from1x3Mat(cvl::Vector3<T> m){
     cv::Mat1d o(3,1);
     for(int i=0;i<3;++i)
         o(i)= m(i);
     return o;
 }

 template<class T>
 cv::Mat1d from3x3Mat(cvl::Matrix3<T> m){
     cv::Mat1d o(9,1);
     for(int i=0;i<9;++i)
         o(i)= m(i);
     return o;
 }

template<class T,class T1>  inline
void conv(const std::vector<cv::Point_<T1> >& in,
          std::vector<cvl::Vector2<T> >& out){
    out.clear();
    out.reserve(in.size());
    for(uint i=0;i<in.size();++i){
        out.push_back(cvl::Vector2<T>(in[i].x,in[i].y));
    }
}

template<class T,typename T1> inline
void conv(const std::vector<cvl::Vector3<T1> >& in,
          std::vector<cv::Point3_<T> >& out){
    out.clear();
    out.reserve(in.size());
    for(uint i=0;i<in.size();++i){
        out.push_back(cv::Point3_<T>(in[i].x,in[i].y,in[i].z));
    }
}
template<class T,typename T1> inline
void conv(const std::vector<cv::Point3_<T> >& in,
          std::vector<cvl::Vector3<T1> >& out){
    out.clear();
    out.reserve(in.size());
    for(uint i=0;i<in.size();++i){
        out.push_back(cvl::Vector3<T1>(in[i].x,in[i].y,in[i].z));
    }
}
template<class T,typename T1> inline
void conv(const std::vector<cvl::Vector2<T1> >& in,
          std::vector<cv::Point_<T> >& out){
    out.clear();
    out.reserve(in.size());
    for(uint i=0;i<in.size();++i){
        out.push_back(cv::Point_<T>(in[i][0],in[i][1]));
    }
}


template<class FROM,class TO>  cv::Mat_<TO> convertTU(const cv::Mat_<FROM>& m, double scale=1){
    cv::Mat_<TO> out(m.rows,m.cols);
    for(int r=0;r<m.rows;++r)
        for(int c=0;c<m.cols;++c)
            out(r,c)=TO(((double)m(r,c))*scale);
    return out;
}
template<class FROM,class TO>  void convertTU(const cv::Mat_<FROM>& m,cv::Mat_<TO>& out,double scale=1){
    out=convertTU<FROM,TO>(m,scale);
}

template<class TO>

cv::Mat_<TO> rgb2gray(const cv::Mat_<cv::Vec3b>& m){
    // so if its a sRGB we should do gamma correct and color weighting

    cv::Mat_<TO> out(m.rows,m.cols);
    for(int r=0;r<m.rows;++r)
        for(int c=0;c<m.cols;++c){
            cv::Vec3b v=m(r,c);
            out(r,c)=((float)v[0] + (float)v[1] + (float)v[2])/(3.0f*255.0f);
        }
    return out;
}


}
// end namespace cvl
