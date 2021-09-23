#include <mlib/datasets/stereo_datasets.h>
#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/datasets/daimler/dataset.h>
#include <mlib/datasets/hilti/dataset.h>
namespace cvl {


std::shared_ptr<StereoSequence> kitti_sequence(int sequence, std::string path) {
    return kitti::dataset(path).sequences()[sequence];
}

std::shared_ptr<StereoSequence> daimler_sequence(std::string path, std::string gt_path){
    return daimler::dataset(path,gt_path).sequences()[0];
}
std::shared_ptr<StereoSequence> hilti_sequence(int sequence, std::string path){
    return hilti::dataset(path).sequence(sequence);
}

std::shared_ptr<StereoSequenceStream> kitti_sequence_stream(int sequence, std::string path){
    return std::make_shared<StereoSequenceStream> (kitti_sequence(sequence, path));
}
std::shared_ptr<StereoSequenceStream> daimler_sequence_stream(std::string path, std::string gt_path){
    return std::make_shared<StereoSequenceStream> (daimler_sequence(path, gt_path));
}
std::shared_ptr<StereoSequenceStream> hilti_sequence_stream(int sequence, std::string path){
    return std::make_shared<StereoSequenceStream> (hilti_sequence(sequence, path));
}

std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>> buffered_kitti_sequence(int sequence, int offset, std::string path){
    return std::make_shared<BufferedStream<StereoSequenceStream>>(offset, kitti_sequence_stream(sequence, path));
}
std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>> buffered_daimler_sequence(int offset, std::string path, std::string gt_path){
    return std::make_shared<BufferedStream<StereoSequenceStream>>(offset, daimler_sequence_stream(path, gt_path));
}
std::shared_ptr<cvl::BufferedStream<StereoSequenceStream>> buffered_hilti_sequence(int sequence, int offset, std::string path){
    return std::make_shared<BufferedStream<StereoSequenceStream>>(offset, hilti_sequence_stream(sequence, path));
}

} // end namespace cvl
