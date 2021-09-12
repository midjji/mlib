#include <mlib/datasets/stereo_datasets.h>
#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/datasets/daimler/dataset.h>
namespace cvl {



std::shared_ptr<StereoSequence>
kitti_sequence(int sequence,
        std::string path) {
    return kitti::dataset(path).sequences()[sequence];
}
cvl::BufferedStream<StereoSequence> buffered_kitti_sequence(
        int sequence,
        int offset,
        std::string path){
return cvl::BufferedStream<StereoSequence>(offset, kitti_sequence(sequence,path));
}

namespace  daimler{


const DaimlerDataset& dataset(std::string path, std::string gt_path)
{
    // magic static, thread safe as of C++11, mostly, always for 17?
    static DaimlerDataset ds(path,gt_path);
    return ds;
}
}
std::shared_ptr<StereoSequence>
daimler_sequence(std::string path, std::string gt_path){
    return daimler::dataset(path,gt_path).sequences()[0];
}



cvl::BufferedStream<StereoSequence> buffered_daimler_sequence(
        int offset,
        std::string path, std::string gt_path)
{
    return cvl::BufferedStream<StereoSequence>(offset, daimler_sequence(path,gt_path));
}

}
