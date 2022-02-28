#include <opencv2/highgui.hpp>
#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/datasets/daimler/dataset.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/datasets/stereo_datasets.h>
#include <mlib/cuda/klt/internal/texture.h>


using namespace cvl;
using std::cout;using std::endl;

using namespace kitti;

void filter(cv::Mat1f im)
{
    Texture<float,false> host_image(im);
    Texture<float,true> dev_image=im;



}


void inspect_stereo_sequence(std::shared_ptr<StereoSequence> seq)
{
    std::string name=seq->name();
    for(int i=0;i<seq->samples();i+=1)
    {

        auto sample  = seq->stereo_sample(i);
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
    //inspect_stereo_dataset(new DaimlerDataset("/storage/datasets/daimler/2020-04-26/08/"));
    //inspect_stereo_dataset(new KittiDataset("/home/mikael/datasets/kitti/odometry/"));
    auto it=buffered_kitti_sequence();
    auto sample=it->next();
    std::string name;
    while((sample=it->next())){
        imshow(sample->rgb(0),name + " Left");
        imshow(sample->rgb(0),name + " Right");
        imshow(sample->display_disparity(),name+" Disparity");
        cv::waitKey(0);
    }

    buffered_daimler_sequence();










}
