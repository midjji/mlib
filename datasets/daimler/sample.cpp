#include <mlib/datasets/daimler/sample.h>
#include <mlib/utils/mlog/log.h>

#include <mlib/utils/cvl/triangulate.h>
#include <mlib/datasets/stereo_calibration.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/utils/colormap.h>
#include <mlib/utils/bounding_box.h>
namespace cvl {

int DaimlerSample::rows() const{    return 1024;}
int DaimlerSample::cols()const{return 2048;}
int DaimlerSample::frame_id()const{return frameid_;}
int DaimlerSample::sequence_id() const{return 0;}
double DaimlerSample::time() const{    return time_;}
cv::Mat1f DaimlerSample::disparity_image() const{return disparity_;}
cv::Mat3b DaimlerSample::rgb(int id) const {    return image2rgb3b(images.at(id),1.0/16.0);}
cv::Mat1b DaimlerSample::grey1b(int id) const{    return image2grey1b(images.at(id),1.0/16.0);}
cv::Mat1w DaimlerSample::grey1w(int id) const{    return images.at(id).clone();}
cv::Mat1f DaimlerSample::grey1f(int id) const{    return image2grey1f(images.at(id));}
cv::Mat3f DaimlerSample::rgb3f(int id) const{    return image2rgb3f(images.at(id),1.0/16.0);}
float DaimlerSample::disparity_impl(double row, double col) const{
    return disparity_(std::round(row),std::round(col));
}









DaimlerSample::DaimlerSample(std::vector<cv::Mat1w> images,
                             cv::Mat1f disparity_,
                             cv::Mat1b labels,
                             int frameid, double time_):images(images), disparity_(disparity_),
    labels(labels), frameid_(frameid),time_(time_){}





bool DaimlerSample::is_car(Vector2d rowcol) const{    return is_car(rowcol(0),rowcol(1));}


bool DaimlerSample::is_car(double row, double col) const{

    assert(row>=0);
    assert(col>=0);
    assert(row<labels.rows);
    assert(col<labels.cols);
    return labels(int(std::round(row)),int(std::round(col)))>0;
} // is disparity still offset by one frame? check !

cv::Mat3b DaimlerSample::show_labels() const{
    cv::Mat3b im(rows(), cols());
    for(int r=0;r<rows();++r)
        for(int c=0;c<cols();++c){
            int label=labels(r,c);
            mlib::Color color=mlib::Color::nrlarge(label);
            im(r,c)=cv::Vec3b(color.getB(),color.getG(), color.getR());
        }
    return im;
}

}

