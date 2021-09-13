
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mlib/opencv_util/cv.h"


#include "mlib/utils/simulator_helpers.h"
#include "mlib/utils/colormap.h"
#include "mlib/utils/string_helpers.h"

#include "mlib/utils/cvl/syncque.h"
#include <mlib/utils/cvl/convertopencv.h>


using std::cout;
using std::endl;
using namespace cvl;
namespace mlib{

double abs(const cv::Point2f& p){
    return sqrt(p.x*p.x + p.y*p.y);
}
double dist(const cv::Point2f& a,const cv::Point2f& b){
    return sqrt((a.x -b.x)*(a.x -b.x) + (a.y -b.y)*(a.y -b.y));
}
double sqdist(const cv::Point2f& a,const cv::Point2f& b){
    return ((a.x -b.x)*(a.x -b.x) + (a.y -b.y)*(a.y -b.y));
}
double l1dist(const cv::Point2f& a,const cv::Point2f& b){
    return fabs(a.x -b.x) + fabs(a.y -b.y);
}

cv::Mat3b drawCount(std::vector<double> vals){
    int rows=1024;
    int cols=1024;
    cv::Mat3b img(rows,cols);
    if(vals.size()<2)
        return img;

    for(int i=0;i<rows*cols*3;++i)
        img.data[i]=(unsigned char)0;


    std::vector<int> hist;hist.resize(cols,0);

    sort(vals.begin(),vals.end());
    int vminv=int(vals.at(int(round((int(vals.size())-1)*0.000))));
    int vmaxv=int(vals.at(int(round((int(vals.size())-1)*1))));
    for(double  v:vals){

        int index= int(round((cols-1)*(v - vminv )/(vmaxv-vminv)));
        if(vmaxv==v)
            index=512;
        hist.at(index)+=1;
    }
    int maxv=hist.at(0);
    int minv=hist.at(0);
    for(int v:hist){
        if(v>maxv)
            maxv=v;
        if(v<minv)
            minv=v;
    }
    cv::putText(img,"min: "+toZstring(vminv),cv::Point2f(100,100),0,1,convert(Color::blue()),2);
    cv::putText(img,"max: "+toZstring(vmaxv),cv::Point2f(100,150),0,1,convert(Color::blue()),2);
    cv::putText(img,"count min: "+toZstring(minv),cv::Point2f(100,200),0,1,convert(Color::blue()),2);
    cv::putText(img,"count max: "+toZstring(maxv),cv::Point2f(100,250),0,1,convert(Color::blue()),2);

    for(int i=0;i<cols;++i)
        cv::line(img,cv::Point2i(i,1024),cv::Point2i(i,int(1024-hist.at(i)*1024.0/maxv)),cv::Scalar(255,0,0),1);
    return img;
}


std::vector<Vector2d> conv(const std::vector<cv::KeyPoint>& kps){
    std::vector<Vector2d> ret;ret.reserve(kps.size());
    for(const cv::KeyPoint& kp:kps)
        ret.push_back(Vector2d(kp.pt.x,kp.pt.y));
    return ret;
}

std::vector<std::vector<Vector2d> > conv(const std::vector<std::vector<cv::KeyPoint> >& kps){
    std::vector<std::vector<Vector2d>> ret;ret.reserve(kps.size());
    for(const std::vector<cv::KeyPoint>& kp:kps)
        ret.push_back(conv(kp));
    return ret;
}


void drawCircle(cv::Mat3b im,
                cvl::Vector2d center,
                Color color,
                float radius,
                float thickness){
    int row=int(std::round(center[0]));
    int col=int(std::round(center[1]));
    // cv drawings generally appear safe for out of bounds drawing
    cv::circle(im,cv::Point2i(col,row),int(radius), color.toScalar<cv::Scalar>(),int(thickness));
}




cv::Mat getSubMat(cv::Mat im,uint col, uint row, uint width, uint height){
    return  im(cv::Rect(col,row,width,height));
}


cv::Mat smooth(cv::Mat im,double sigma){
    cv::Mat tmp;
    cv::GaussianBlur(im, tmp, cv::Size(0, 0), sigma); // gaussian blurr works ok but likely better with an edgepreserving smoother...
    return tmp;
}
cv::Mat resize(cv::Mat im,double factor){

    cv::Mat tmp;
    cv::resize(im,tmp,cv::Size(),factor,factor);
    return tmp;
}
cv::Mat resize(cv::Mat im, int rows, int cols){
    cv::Mat tmp;
    cv::resize(im,tmp,cv::Size(cols,rows));
    return tmp;
}
cv::Scalar convert(Color color){
    return cv::Scalar(color.getB(),color.getG(),color.getR());
}




void getMagnitudeGradient(cv::Mat1f im,cv::Mat1f& magn ){


    cv::Mat1f drow=cv::Mat1f(im.rows,im.cols);
    cv::Mat1f dcol=cv::Mat1f(im.rows,im.cols);
    cv::Scharr(im,drow,im.depth(),0,1,1.0f/32.0f);
    cv::Scharr(im,dcol,im.depth(),1,0,1.0f/32.0f);
    magn=cv::Mat1f(im.rows,im.cols);
    for(int row=0;row<magn.rows;++row)
        for(int col=0;col<magn.cols;++col)
            magn(row,col)=std::abs(drow(row,col)) + std::abs(dcol(row,col));

    cv::blur(magn,magn,cv::Size(3,3));

}

void getGradients(cv::Mat1f im,cv::Mat1f& drow,cv::Mat1f& dcol, cv::Mat1f& drowcol ){


    drow=cv::Mat1f(im.rows,im.cols);
    dcol=cv::Mat1f(im.rows,im.cols);
    drowcol=cv::Mat1f(im.rows,im.cols);

    // dy for cv is dcol and dx is drow
    cv::Scharr(im,drow,im.depth(),0,1,1.0f/32.0f);
    cv::Scharr(im,dcol,im.depth(),1,0,1.0f/32.0f);
    cv::Scharr(drow,drowcol,im.depth(),1,0,1.0f/32.0f);
    // dcol then drow or drow dcol should not matter...
    // is it that there is too much lp?
    //drowcol=-drowcol;
    //drowcol=-drowcol*4;





#if 0
    {
        cv::Mat1f kx,ky;
        cv::getDerivKernels(kx,ky,1,0,CV_SCHARR,true);
        cv::sepFilter2D(im,drow,im.depth(),kx,ky);
    }
    {
        cv::Mat1f kx,ky;
        cv::getDerivKernels(kx,ky,0,1,CV_SCHARR,true);
        cv::sepFilter2D(im,dcol,im.depth(),kx,ky);
    }
    {
        cv::Mat1f kx,ky;
        cv::getDerivKernels(kx,ky,1,1,CV_SCHARR,true);
        cv::sepFilter2D(im,drowcol,im.depth(),kx,ky);
    }
#endif
}
void increaseContrast(cv::Mat1f& im){
    float minv,maxv;
    minmax(im,minv,maxv);
    cout<<"minmax:"<<minv<<", "<<maxv<<endl;
    for(int row=0;row<im.rows;++row)
        for(int col=0;col<im.cols;++col){
            im(row,col)=(im(row,col)-minv)/(maxv-minv);
        }
}

}// end namespace mlib



