#include <mlib/datasets/kitti/odometry/sample.h>
namespace cvl {
namespace kitti {
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
}
KittiOdometrySample::KittiOdometrySample(std::vector<cv::Mat1w> images,
                    cv::Mat1f disparity_,
                    int sequenceid_,
                    int frameid_, double time_):
    images(images),
    disparity_(disparity_),
    sequenceid_(sequenceid_),
    frameid_(frameid_), time_(time_)
{}
int KittiOdometrySample::frameid() const{return frameid_;}
int KittiOdometrySample::sequenceid() const{return sequenceid_;}
int KittiOdometrySample::rows() const{return disparity_.rows;}
int KittiOdometrySample::cols() const{return disparity_.cols;}
double KittiOdometrySample::time() const{return time_;}
float KittiOdometrySample::disparity(double row, double col) const {
    if(row<0) return -2.0f;
    if(col<0) return -3.0f;
    if(rows()<=row) return -4.0f;
    if(cols()<=col) return -5.0f;
    if(std::isnan(row+col)) return -6.0f;
    return disparity_(int(std::round(row)),int(std::round(col)));
}
cv::Mat3b KittiOdometrySample::rgb(uint id){
return convertw2rgb8(images.at(id));
}
cv::Mat1b KittiOdometrySample::greyb(uint id){
    auto& in=images.at(id);

    cv::Mat1b out(rows(),cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            auto v=in(r,c);
            if(v>255)v=255;
            out(r,c)=uchar(v);
        }
    return out;
}
cv::Mat1w KittiOdometrySample::greyw(uint id){
    return images.at(id).clone();
}
cv::Mat1f KittiOdometrySample::greyf(uint id){
    auto& in=images.at(id);

    cv::Mat1f out(rows(),cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            out(r,c)=float(in(r,c))/16.0F;
        }
    return out;
}
cv::Mat1b KittiOdometrySample::disparity_image() const{
    cv::Mat1b im(rows(), cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            float disp=disparity(r,c);
            disp*=2;
            if(disp<0) disp=0;
            if(disp>255)
                disp=255;
            im(r,c)=uint8_t(disp);
        }
    return im;
}
cv::Mat1f KittiOdometrySample::disparity_imagef() const{
    return disparity_.clone();
}
}

}
