#include <mlib/datasets/daimler/sample.h>
#include <mlib/utils/mlog/log.h>

#include <mlib/utils/cvl/triangulate.h>
#include <mlib/datasets/stereo_calibration.h>
#include <mlib/opencv_util/convert.h>
#include <mlib/utils/colormap.h>
#include <mlib/utils/bounding_box.h>
namespace cvl {










DaimlerSample::DaimlerSample(float128 time_,const StereoSequence* ss,
                             int frame_id, std::vector<cv::Mat1f> images, cv::Mat1f disparity_,
                             cv::Mat1b labels):
    StereoSample(time_,ss, frame_id, images,disparity_),
    labels(labels){}

DaimlerSample::DaimlerSample(std::vector<cv::Mat1w> images,
              cv::Mat1f disparity_,
              cv::Mat1b labels,
              int frameid, double time_,const StereoSequence* ss):
    StereoSample(time_,ss, frameid,images2grey1f(images),disparity_),labels(labels){}



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

