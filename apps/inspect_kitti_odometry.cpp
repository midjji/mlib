#include <mlib/utils/files.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/simulator_helpers.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <mlib/datasets/kitti/odometry/kitti.h>
#include <mlib/opencv_util/cv.h>
#include <mlib/utils/cvl/convertopencv.h>
#include <mlib/cuda/edgels.h>
using namespace mlib;
using namespace cvl;
using std::cout;using std::endl;

using namespace kitti;




int main()
{
    KittiDataset kd("/home/mikael/datasets/kitti/odometry/");
    kd.init();
    while(true){
        for(uint seq=0;seq<kd.sequences.size();++seq){
            Sequence sq=kd.getSequence(seq);
            for(int i=0;i<sq.images;i+=1){


                std::vector<cv::Mat1w> imgs;
                cv::Mat1f disp;
                if(!sq.getImages(imgs,disp,i)) continue;
                for(int r=0;r<disp.rows;++r)
                    for(int c=0;c<disp.cols;++c){
                        float v=disp(r,c);if(v<-1) v=-1;
                        v+=64;
                        disp(r,c)=v;

                    }

                disp(0,0)=256-64;
                cv::Mat1f L;
                convertTU(imgs[0],L);
                getGoodEdgels(L,sq.getK(),sq.getPose(i+1)*sq.getPose(i).inverse());

                cv::imshow("Kitti Left",normalize01(imgs[0]));
                cv::imshow("Kitti Right",normalize01(imgs[1]));
                cv::imshow("Disparity", normalize01(disp));
                cv::waitKey(0);

            }

        }
    }

    // epiline thingy




}
