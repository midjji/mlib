#pragma once
#include <mlib/datasets/stereo_dataset.h>
#include <mlib/utils/buffered_dataset_stream.h>
namespace cvl {

std::shared_ptr<StereoSequence> kitti_sequence(int sequence=0, std::string path="/storage/datasets/kitti/odometry/");
std::shared_ptr<StereoSequence> daimler_sequence(std::string path="/storage/datasets/daimler/2020-04-26/08", std::string gt_path="");
std::shared_ptr<StereoSequence> hilti_sequence(int sequence=0, std::string path="/storage/datasets/hilti/preprocessed/Construction_Site_1/");


struct StereoSequenceStream{
    using sample_type=std::shared_ptr<StereoSample>;
    StereoSequenceStream(std::shared_ptr<StereoSequence> ss):ss(ss){}
    std::shared_ptr<StereoSequence> ss;
    std::shared_ptr<StereoSample> sample(int index) const{return ss->stereo_sample(index);}
    int samples(){return ss->samples();}
};



std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>>
buffered_kitti_sequence(
        int sequence=0,
        int offset=0,
        std::string path="/storage/datasets/kitti/odometry/");

std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>>
buffered_daimler_sequence(
        int offset=0,
        std::string path="/storage/datasets/daimler/2020-04-26/08",
        std::string gt_path="");

std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>>
buffered_hilti_sequence(int sequence=0,
        int offset=0,
        std::string path="/storage/datasets/hilti/preprocessed/");
}

