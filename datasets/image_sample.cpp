#include <mlib/datasets/image_sample.h>
#include <mlib/opencv_util/convert.h>

namespace cvl{
ImageSample::ImageSample(float128 time,const StereoSequence* ss, int frame_id_,std::vector<cv::Mat1f> images):
    Sample(time,ss),
    frame_id_(frame_id_),
    images(images){}
ImageSample::~ImageSample(){}

cv::Mat3b ImageSample::rgb(int id) const {    return image2rgb3b(grey1f(id),1.0/16.0);}
cv::Mat1b ImageSample::grey1b(int id) const{    return image2grey1b(grey1f(id),1.0/16.0);}
cv::Mat1f ImageSample::grey1f(int id) const{    return images.at(id).clone();}

int ImageSample::type() const{
    return Sample::image;
}

int ImageSample::rows() const{
    return images[0].rows;
}
int ImageSample::cols() const{
    return images[0].cols;
}
int ImageSample::frame_id() const{return frame_id_;}
}
