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
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/colormap.h>
#include <mlib/opencv_util/type2str.h>


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


void abs_inplace(cv::Mat1f& image);
void abs_inplace(cv::Mat1d& image);




void draw_circle(cv::Mat3b& im,
                cvl::Vector2d center /*row,col*/,
                Color color=Color::cyan(),
                float radius=5,
                float thickness=2);

cv::Mat getSubMat(cv::Mat im,uint col, uint row, uint width, uint height);
cv::Mat smooth(cv::Mat im,double sigma=1);
cv::Mat resize(cv::Mat im,double factor);
cv::Mat resize(cv::Mat im,int rows, int cols);

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
    T min,max;
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





std::string str(cv::Mat m);



}// end namespace mlib



