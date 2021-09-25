#pragma once
#include <memory>

#include <mlib/datasets/stereo_dataset.h>
#include <mlib/datasets/daimler/sample.h>


namespace cvl{
namespace mtable{
struct GTDB;
}
class DaimlerSequence:public StereoSequence
{

    std::weak_ptr<DaimlerSequence> wself;
    DaimlerSequence(std::string dataset_path, std::string gt_path="");
        double framerate() const override;
public:
     int samples() const override;
     std::shared_ptr<StereoSample> stereo_sample(int index) const override;
     std::shared_ptr<DaimlerSample> sample(int index) const;

     int rows() const override;
     int cols() const override;
     std::string name() const override;
     StereoCalibration calibration() const override;
     std::shared_ptr<Frameid2TimeMap> fid2time() const override;
     std::vector<double> times() const override;

     std::vector<PoseD> gt_poses() const override;





    static std::shared_ptr<DaimlerSequence> create(std::string path, std::string sequence_name);

    std::string path;
    std::shared_ptr<mtable::GTDB> gt_storage;


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
