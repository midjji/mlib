#pragma once

#include <mlib/datasets/stereo_dataset.h>
#include <mlib/datasets/daimler/sample.h>
#include <mlib/datasets/daimler/database.h>

namespace cvl{

class DaimlerSequence:public StereoSequence
{

public:
     int samples() const override;
     std::shared_ptr<StereoSample> sample(int index) const override;
     int rows() const override;
     int cols() const override;
     std::string name() const override;
     StereoCalibration calibration() const override;
     std::shared_ptr<Frameid2TimeMap> fid2time() const override;
     int sequence_id() const override;
     std::vector<PoseD> gt_poses() const override;




    using sample_type=std::shared_ptr<DaimlerSample>;

    DaimlerSequence(std::string dataset_path, std::string gt_path="");
    std::string path;
    mtable::gt_db_type gt_storage;
    std::shared_ptr<DaimlerSample> get_sample(uint index) const;

    double fps() const;

    std::vector<PoseD> gt_poses_; // interface..


private:
    uint total_samples=0;
    cv::Mat1b get_cars(cv::Mat1b labels) const;
    bool read_images(uint sampleindex,
                     cv::Mat1w& left,
                     cv::Mat1w& right,
                     cv::Mat1b& labels,
                     cv::Mat1f& disparity) const;



    bool read_right=true;
};





} // end namespace daimler
