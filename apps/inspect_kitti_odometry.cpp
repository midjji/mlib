#include <opencv2/highgui.hpp>
#include <mlib/datasets/kitti/odometry/kitti.h>

#include <mlib/opencv_util/imshow.h>


namespace cvl {


namespace kitti {


void inspect(){
    KittiDataset kd("/home/mikael/datasets/kitti/odometry/");

    while(true){
        for(uint seq=0;seq<kd.sequences().size();++seq){
            std::shared_ptr<kitti::Sequence> sq=kd.getSequence(seq);
            std::string name=sq->name();
            for(int i=0;i<sq->samples();i+=1){
                auto sample=sq->sample(i);

                imshow(sample->rgb(0),name + " Left");
                imshow(sample->rgb(0),name + " Right");
                imshow(sample->display_disparity(),name+" Disparity");
                cv::waitKey(0);

            }

        }
    }
}
}
}


int main()
{
    cvl::kitti::inspect();
}
