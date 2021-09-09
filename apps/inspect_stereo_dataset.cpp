#include <opencv2/highgui.hpp>
#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/datasets/daimler/dataset.h>
#include <mlib/opencv_util/imshow.h>


using namespace cvl;
using std::cout;using std::endl;

using namespace kitti;

void inspect_stereo_sequence(const std::shared_ptr<StereoSequence>& seq){
    std::string name=seq->name();
    for(int i=0;i<seq->samples();i+=1)
    {

        auto sample  = seq->sample(i);
        imshow(sample->rgb(0),name + " Left");
        imshow(sample->rgb(0),name + " Right");
        imshow(sample->display_disparity(),name+" Disparity");
        cv::waitKey(0);

    }
}
void inspect_stereo_dataset(StereoDataset* dataset){
    auto& kd =*dataset;

        for(const auto& seq:kd.sequences())
        {
            inspect_stereo_sequence(seq);

        }
}


int main()
{
    inspect_stereo_dataset(new DaimlerDataset("/storage/datasets/daimler/2020-04-26/08/"));
    inspect_stereo_dataset(new KittiDataset("/home/mikael/datasets/kitti/odometry/"));








}
