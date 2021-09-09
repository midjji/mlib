#include <iostream>
#include <mlib/datasets/kitti/mots/dataset.h>
#include <opencv4/opencv2/highgui.hpp>
#include <mlib/opencv_util/imshow.h>

//using namespace mlib;
using namespace cvl;
using namespace std;




int main()
{

    KittiMotsDataset kmd("/home/mikael/datasets/kitti/mots/");



    bool training=true;
    int seq=0;
    for(int i=0;i<kmd.samples(training, seq);++i){
        auto s=kmd.get_sample(training,seq,i);
        cvl::imshow("rgb left",s->rgb(0));
        cvl::imshow("rgb right",s->rgb(1));
        cv::waitKey(0);
    }
}
