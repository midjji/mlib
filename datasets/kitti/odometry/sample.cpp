#include <mlib/datasets/kitti/odometry/sample.h>
#include <mlib/opencv_util/convert.h>
namespace cvl {
namespace kitti {

KittiOdometrySample::KittiOdometrySample(std::vector<cv::Mat1w> images,
                                         cv::Mat1f disparity_,
                                         int sequenceid_,
                                         int frameid_, double time_):
    images(images),
    disparity_(disparity_),
    sequenceid_(sequenceid_),
    frameid_(frameid_), time_(time_){}

int KittiOdometrySample::frame_id() const{return frameid_;}
int KittiOdometrySample::sequence_id() const{return sequenceid_;}
int KittiOdometrySample::rows() const{return disparity_.rows;}
int KittiOdometrySample::cols() const{return disparity_.cols;}
double KittiOdometrySample::time() const{return time_;}

cv::Mat3b KittiOdometrySample::rgb(int id) const{
    return image2rgb3b(images.at(id),1.0F/16.0F);
}
cv::Mat3f KittiOdometrySample::rgb3f(int id) const{
    return image2rgb3f(images.at(id),1.0F/16.0F);
}
cv::Mat1f KittiOdometrySample::grey1f(int id) const{
    return image2grey1f(images.at(id),1.0F/16.0F);
}
cv::Mat1w KittiOdometrySample::grey1w(int id) const{
    return images.at(id).clone();
}
cv::Mat1b KittiOdometrySample::grey1b(int id) const{
    return image2grey1b(images.at(id),1.0F/16.0F);
}
cv::Mat1f KittiOdometrySample::disparity_image() const
{
    return disparity_;
}
float KittiOdometrySample::disparity_impl(double row, double col) const{
    return disparity_(std::round(row),std::round(col));
}
}

}
