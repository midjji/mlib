#pragma once
#include <map>
#include <opencv2/core/mat.hpp>
#include <mlib/utils/cvl/pose.h>

#include <mlib/datasets/stereo_sample.h>
#include <mlib/utils/imu_data.h>


namespace cvl{
namespace hilti {



/**
 * @brief The ImageSample class
 */
struct HiltiImageSample:public StereoSample
{        
    // we read the rectified images, nr 0 is left, nr 1 is right, 2-4 are cam2-4 and 5 is disparity
    HiltiImageSample(
            float128 time,
            const StereoSequence* ss,
            int  frame_id,
            std::map<int,cv::Mat1f> images, // nr 5 is disparity between left and right
            std::vector<imu::Data> imu_datas);

    virtual cv::Mat1f grey1f(int i) const override;
    bool complete() const;
    bool stereo() const;
    bool has(int i) const;
    virtual int type() const override;
    void show() const;
private:


    // the raw images
    // not all are guaranteed to exist?
    std::map<int,cv::Mat1f> images;
    std::vector<imu::Data> imu_datas;
};



}
}
