#include <mlib/datasets/kitti/odometry/sample.h>
#include <mlib/opencv_util/convert.h>
namespace cvl {
namespace kitti {

KittiOdometrySample::KittiOdometrySample(float128 time,const StereoSequence* ss,
                                         int frame_id, std::vector<cv::Mat1f> images,
                                         cv::Mat1f disparity_):
    StereoSample(time,ss,  frame_id, images,disparity_){}

KittiOdometrySample::KittiOdometrySample(std::vector<cv::Mat1w> images,
                    cv::Mat1f disparity_,
                    int frame_id_, double time_,const StereoSequence* ss):
    StereoSample( time_,ss, frame_id_,images2grey1f(images),disparity_)
{

}

}

}
