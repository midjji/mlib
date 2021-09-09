#include <mlib/opencv_util/convert.h>

namespace cvl {
cv::Mat3b image2rgb3b(const cv::Mat1b& img,
               float scale,
               float offset)
{
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            if(tmp<0) tmp=0;
            if(tmp>255) tmp=255;
            rgb(r,c)=cv::Vec3b(tmp,tmp,tmp);
        }
    return rgb;
}
cv::Mat3b image2rgb3b(const cv::Mat1w& img,
               float scale,
               float offset)
{
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            if(tmp<0) tmp=0;
            if(tmp>255) tmp=255;
            rgb(r,c)=cv::Vec3b(tmp,tmp,tmp);
        }
    return rgb;
}
cv::Mat3b image2rgb3(const cv::Mat1f& img,
               float scale,
                     float offset){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            if(tmp<0) tmp=0;
            if(tmp>255) tmp=255;
            rgb(r,c)=cv::Vec3b(tmp,tmp,tmp);
        }
    return rgb;
}

cv::Mat3f image2rgb3f(const cv::Mat1w& img,
                      float scale,
                      float offset){
    cv::Mat3f rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            rgb(r,c)=cv::Vec3f(tmp,tmp,tmp);
        }
    return rgb;
}
cv::Mat1f image2grey1f(const cv::Mat1w& img,
                      float scale,
                       float offset){
    cv::Mat1f rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            rgb(r,c)=tmp;
        }
    return rgb;
}
cv::Mat1b image2grey1b(const cv::Mat1w& img,
                      float scale,
                       float offset){
    cv::Mat1b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c)*scale + offset;
            rgb(r,c)=tmp;
        }
    return rgb;
}

}
