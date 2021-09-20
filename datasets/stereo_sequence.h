#pragma once
#include <mlib/datasets/stereo_sample.h>
#include <mlib/datasets/stereo_calibration.h>
#include <mlib/datasets/frameid2time.h>
#include <mlib/datasets/sequence.h>
namespace cvl {

class StereoSequence{
public:
    using sample_type=std::shared_ptr<StereoSample>;
    virtual ~StereoSequence();
    virtual int samples() const=0;
    virtual std::shared_ptr<StereoSample> sample(int index) const=0;
    virtual int rows() const=0;
    virtual int cols() const=0;
    virtual std::string name() const=0;
    virtual StereoCalibration calibration() const=0;
    virtual std::shared_ptr<Frameid2TimeMap> fid2time() const =0;
    virtual std::vector<double> times() const=0;
    virtual int id() const;
    virtual std::vector<PoseD> gt_poses() const;
    std::vector<PoseD> gt_vehicle_poses() const;

    //int index=0;
    //std::shared_ptr<StereoSample> next();



};
}
