#include <opencv2/highgui.hpp>
#include <mlib/datasets/kitti/odometry/kitti.h>

#include <mlib/opencv_util/imshow.h>

using namespace cvl;
using std::cout;using std::endl;

using namespace kitti;




int main()
{
    KittiDataset kd("/home/mikael/datasets/kitti/odometry/");

    while(true){
        for(uint seq=0;seq<kd.sequences().size();++seq){
            Sequence sq=*kd.getSequence(seq);
            std::string name=sq.name();
            for(int i=0;i<sq.samples();i+=1){
                auto sample=sq.get_sample(i);

                imshow(sample->rgb(0),name + " Left");
                imshow(sample->rgb(0),name + " Right");
                imshow(sample->display_disparity(),name+" Disparity");
                cv::waitKey(0);

            }

        }
    }
}
