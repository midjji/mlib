#include <daimler/sample.h>
#include <mlib/utils/mlog/log.h>

#include <mlib/utils/cvl/triangulate.h>
#include <daimler/calibration.h>
using std::cout;
using std::endl;
namespace cvl {

namespace {
cv::Mat3b convertw2rgb8(cv::Mat1w img){
    cv::Mat3b rgb(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            uint16_t tmp=img(r,c)/16;
            if(tmp>255)tmp=255;
            rgb(r,c)=cv::Vec3b(tmp,tmp,tmp);
        }
    return rgb;
}
cv::Mat1b convertw2gray8(cv::Mat1w img){
    cv::Mat1b gray(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            uint16_t tmp=img(r,c)/16;
            if(tmp>255)tmp=255;
            gray(r,c)=uint8_t(tmp);
        }
    return gray;
}
cv::Mat1f convertw2grayf(cv::Mat1w img){
    cv::Mat1f gray(img.rows,img.cols);
    for(int r=0;r<img.rows;++r)
        for(int c=0;c<img.cols;++c){
            float tmp=img(r,c);
            gray(r,c)=tmp;
        }
    return gray;
}
}

DaimlerSample::DaimlerSample(std::vector<cv::Mat1w> images,
                             cv::Mat1f disparity_,
                             cv::Mat1b labels,
                             int frameid):images(images), disparity_(disparity_),
    labels(labels), frameid_(frameid){}

float DaimlerSample::disparity(double row, double col) const{ // row,col

    if(row<0) return -100.0f;
    if(col<0) return -200.0f;
    if(rows()<=row) return -300.0f;
    if(cols()<=col) return -400.0f;
    if(std::isnan(row+col)) return -500.0f;
    return disparity_(int(std::round(row)),int(std::round(col)));
}
float DaimlerSample::disparity(Vector2d rowcol) const{return disparity(rowcol[0],rowcol[1]);}


Vector3d DaimlerSample::get_3d_point(double row, double col) const
{
    double disp=disparity(row,col);
    if(disp<0) disp=0;
    return DaimlerCalibration::common().triangulate_ray(Vector2d(row,col),disp).dehom();
}
bool DaimlerSample::is_car(Vector2d rowcol) const{
    return is_car(rowcol(0),rowcol(1));
}
bool DaimlerSample::is_car(double row, double col) const{

    assert(row>=0);
    assert(col>=0);
    assert(row<labels.rows);
    assert(col<labels.cols);
    return labels(int(std::round(row)),int(std::round(col)));
} // is disparity still offset by one frame? check !

int DaimlerSample::rows() const{
    return 1024;

}
int DaimlerSample::cols()const{return 2048;}
int DaimlerSample::frameid()const{return frameid_;}

cv::Mat1b DaimlerSample::disparity_image() const{

    cv::Mat1b im(rows(), cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            float disp=disparity_(r,c);
            disp*=2;
            if(disp<0) disp=0;
            if(disp>255)
                disp=255;
            im(r,c)=uint8_t(disp);
        }
    return im;
}
cv::Mat1f DaimlerSample::disparity_imagef() const{
return disparity_;
}
cv::Mat3b DaimlerSample::disparity_image_rgb() const{

    cv::Mat3b im(rows(), cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            float disp=disparity_(r,c);
            disp*=2;
            if(disp<0) disp=0;
            if(disp>255)
                disp=255;
            im(r,c)=cv::Vec3b(1,1,1)*disp;
        }
    return im;
}
cv::Mat3b DaimlerSample::rgb(uint id) const // for visualization, new clone
{
    return convertw2rgb8(images.at(id));
}
cv::Mat1b DaimlerSample::gray(uint id) const // for visualization, new clone
{
    return convertw2gray8(images.at(id));
}
cv::Mat1f DaimlerSample::grayf(uint id) const{
    return convertw2grayf(images.at(id));
}
cv::Mat3b DaimlerSample::show_labels() const{
    cv::Mat3b im(rows(), cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            float disp=labels(r,c);
            im(r,c)=cv::Vec3b(1,1,1)*disp;
        }
    return im;
}
}

