#pragma once
/* ********************************* FILE ************************************/
/** \file    cv.h
 *
 * \brief    This header the ImageShow:: thread safe showing of images using opencv... and various opencv helpers
 *
 * \remark
 * - c++11
 *
 * \todo
 * - improve separation
 *
 *
 *
 * \author   Mikael Persson
 * \date     2013-04-01
 * \note MIT licence
 *
 ******************************************************************************/
/***
 * Convenient but missing opencv functionality
 */



#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/contrib/contrib.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/nonfree/nonfree.hpp>  // would be great to get rid of this one, its just the surf im using && nothing else

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/colormap.h>
#include <mlib/opencv_util/draw_arrow.h>



namespace mlib{


cv::Mat3b drawCount(std::vector<double> val);
std::vector<cvl::Vector2d> conv(const std::vector<cv::KeyPoint>& kps);
std::vector<std::vector<cvl::Vector2d> > conv(const std::vector<std::vector<cv::KeyPoint> >& kps);

double abs(const cv::Point2f& p);
double dist(const cv::Point2f& a,const cv::Point2f& b);
double sqdist(const cv::Point2f& a,const cv::Point2f& b);
double l1dist(const cv::Point2f& a,const cv::Point2f& b);

// image helpers
cv::Mat gridcombine(std::vector<cv::Mat> imgs);
cv::Point2f gridcoord(std::vector<cv::Mat> imgs,int i);
void imshowmany(std::vector<cv::Mat> imgs,std::string name="map");






void drawCircle(cv::Mat3b im,
                cvl::Vector2d center /*row,col*/,
                Color color=Color::cyan(),
                float radius=5,
                float thickness=2);

cv::Mat getSubMat(cv::Mat im,uint col, uint row, uint width, uint height);
cv::Mat smooth(cv::Mat im,double sigma=1);
cv::Mat resize(cv::Mat im,double factor);
template<class T> cv::Mat_<T> resize(cv::Mat_<T> im,uint rows, uint cols){
    cv::Mat tmp;
    cv::resize(im,tmp,cv::Size(cols,rows));
    return tmp;
}
cv::Scalar convert(Color color);


/**
cv::Point min_loc, max_loc
bad idea though, it assumes alot of wierd shit
cv::minMaxLoc(your_mat, &min, &max, &min_loc, &max_loc);
**/
template<class T> bool minmax(const cv::Mat_<T>& im,T& minv, T& maxv){
    assert(im.rows>1);
    assert(im.cols>1);
    minv=maxv=im(0,0);
    if(im.rows==0||im.cols==0)
        return true;

    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c){
            T v=im(r,c);
            if(v>maxv) maxv=v;
            if(v<minv) minv=v;
        }
    return false;
}

template<class T> T min(const cv::Mat_<T>& im){
    T minv,maxv;
    minmax(im,minv,maxv);
    return minv;
}
template<class T> T max(const cv::Mat_<T>& im){
    T minv,maxv;
    minmax(im,minv,maxv);
    return maxv;
}
template<class T>
cv::Mat1f normalize01(const cv::Mat_<T>& im){
    cv::Mat1f ret(im.rows,im.cols);
    T min,max;min=0;max=1;
    minmax(im, min, max);
    //std::cout<<"cv minmax: "<<min<<" "<<max<<std::endl;
    for(int r=0;r<im.rows;++r)
        for(int c=0;c<im.cols;++c)
            ret(r,c)=((float)(im(r,c)-min))/((float)(max-min));
    return ret;
}



/**
     * @brief getGradients
     * @param im
     * @param drow
     * @param dcol
     * @param drowcol
     * the gradients should be per pixel!
     *
     *
     * opencv is truly horrid with this stuff...
     */
void getGradients(cv::Mat1f im,cv::Mat1f& drow,cv::Mat1f& dcol, cv::Mat1f& drowcol );
void getMagnitudeGradient(cv::Mat1f im,cv::Mat1f& magn );
void increaseContrast(cv::Mat1f& im);



template<class T> std::string get_mat_type(cv::Mat m){
    switch(m.type()){
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

template <class T> std::string display_mat(cv::Mat m){
std::stringstream ss;
    ss<<m.rows<<" "<<m.cols<<" "<<m.type()<<" which is: "<<get_mat_type<T>(m);
    return ss.str();
}
}// end namespace mlib



